import os
from unsloth import FastLanguageModel
from datasets import load_dataset
from unsloth import FastLanguageModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get GPU memory info
if device.type == "cuda":
  gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
  print(f"Total GPU Memory: {gpu_memory:.2f} GB")

SYSTEM = (
  "You are a chemistry expert, and you are helping a student to answer the question."
  " Show formulas and units in your answer, and explain the steps to get the answer when relevant."
)


def pick(row, names, default=""):
  for n in names:
    if n in row and row[n] is not None:
      return str(row[n]).strip()
  return default


def to_sft_text(row):
  question = pick(row, ["question", "prompt", "instruction", "input", "problem"])
  answer = pick(row, ["answer", "response", "output", "completion", "solution", "reference_answer"])
  text = f"{SYSTEM}\n\nProblem:\n{question}\n\nAnswer:\n{answer}"
  return {"text": text}


def main():
  print(f"Using device: {device}")
  ds = load_dataset("summykai/chemistry-sft-ultra")
  train_ds = ds["train"]
  print([train_ds[0]] if len(train_ds) > 0 else [])

  sft_ds = train_ds.map(to_sft_text, remove_columns=train_ds.column_names)

  model_name = os.getenv("MODEL_NAME", "microsoft/Phi-4-mini-reasoning")
  output_dir = os.getenv("OUTPUT_DIR", "Chem-PHI-mini-cpu-sft")
  save_steps = int(os.getenv("SAVE_STEPS", "50000"))
  save_total_limit = int(os.getenv("SAVE_TOTAL_LIMIT", "3"))

  os.makedirs(output_dir, exist_ok=True)

  # Load model and tokenizer with unsloth for efficiency
  model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=6000,
    dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    load_in_4bit=device.type == "cuda",
  )

  # Find actual module names
  target_modules_found = set()
  for name, _ in model.named_modules():
    if 'proj' in name.lower():
      target_modules_found.add(name.split('.')[-1])

  print(f"\nFound projection modules: {target_modules_found}")
  target_modules_list = list(target_modules_found)[:2]
  print(f"Using target modules: {target_modules_list}")

  # Apply LoRA using unsloth's built-in support
  model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=target_modules_list,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    use_rslora=True,
  )
  model.print_trainable_parameters()

  # Prepare model for training
  model = FastLanguageModel.for_training(model)

  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  use_cpu = device.type == "cpu"
  fp16 = device.type == "cuda"  # T4 supports fp16 but not bf16
  bf16 = False

  training_args = SFTConfig(
    output_dir=output_dir,
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    logging_steps=10,
    max_steps=6000,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=save_total_limit,
    max_length=6000,
    report_to="none",
    use_cpu=use_cpu,
    fp16=fp16,
    bf16=bf16,
    dataloader_num_workers=4,
    optim="paged_adamw_8bit",
    warmup_steps=100,
    weight_decay=0.01,
    seed=42,
  )

  trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=sft_ds,
  )

  last_checkpoint = get_last_checkpoint(output_dir)
  if last_checkpoint:
    print(f"Resuming from checkpoint: {last_checkpoint}")
  else:
    print("No checkpoint found. Starting fresh training run.")

  trainer.train(resume_from_checkpoint=last_checkpoint)

  # Save the LoRA adapter
  model.save_pretrained(output_dir)
  tokenizer.save_pretrained(output_dir)

  # Optionally merge LoRA weights with base model and save full model
  merged_model = model.merge_and_unload()
  merged_output_dir = f"{output_dir}-merged"
  os.makedirs(merged_output_dir, exist_ok=True)
  merged_model.save_pretrained(merged_output_dir)
  tokenizer.save_pretrained(merged_output_dir)

  hf_token = os.getenv("HF_TOKEN")
  if not hf_token:
    hf_token = input("Hf token please").strip()
  if hf_token:
    # Push LoRA adapter
    model.push_to_hub(output_dir, token=hf_token)
    tokenizer.push_to_hub(output_dir, token=hf_token)
    # Push merged model
    merged_model.push_to_hub(merged_output_dir, token=hf_token)
    tokenizer.push_to_hub(merged_output_dir, token=hf_token)
    print(f"Successfully pushed LoRA adapter to '{output_dir}'.")
    print(f"Successfully pushed merged model to '{merged_output_dir}'.")
  else:
    print("HF_TOKEN not set. Models saved locally only.")


if __name__ == "__main__":
  main()

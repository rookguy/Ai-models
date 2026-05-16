import os
from datasets import load_dataset
from unsloth import FastLanguageModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
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
    max_seq_length=512,
    dtype=torch.float16 if device.type == "cuda" else torch.float32,
    load_in_4bit=device.type == "cuda",
  )

  # Prepare model for kbit training (quantized)
  if device.type == "cuda":
    model = prepare_model_for_kbit_training(model)

  # Configure LoRA for efficient fine-tuning
  lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
  )

  # Apply LoRA to the quantized model
  model = get_peft_model(model, lora_config)
  model.print_trainable_parameters()

  # Prepare model for training
  model = FastLanguageModel.for_training(model)

  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  use_cpu = device.type == "cpu"
  fp16 = device.type == "cuda"
  bf16 = False

  training_args = SFTConfig(
    output_dir=output_dir,
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="steps",
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    max_length=512,
    report_to="none",
    use_cpu=use_cpu,
    fp16=fp16,
    bf16=bf16,
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

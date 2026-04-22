import os
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

torch.set_default_device("cpu")

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
  print("Using device: cpu")
  ds = load_dataset("summykai/chemistry-sft-ultra")
  train_ds = ds["train"]
  print([train_ds[0]] if len(train_ds) > 0 else [])

  sft_ds = train_ds.map(to_sft_text, remove_columns=train_ds.column_names)

  model_name = os.getenv("MODEL_NAME", "microsoft/Phi-4-mini-reasoning")
  output_dir = os.getenv("OUTPUT_DIR", "Chem-PHI-mini-cpu-sft")
  save_steps = int(os.getenv("SAVE_STEPS", "50000"))
  save_total_limit = int(os.getenv("SAVE_TOTAL_LIMIT", "3"))

  os.makedirs(output_dir, exist_ok=True)

  tokenizer = AutoTokenizer.from_pretrained(model_name)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  model = AutoModelForCausalLM.from_pretrained(model_name)
  model.to("cpu")

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
    use_cpu=True,
    fp16=False,
    bf16=False,
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

  model.save_pretrained(output_dir)
  tokenizer.save_pretrained(output_dir)

  hf_token = os.getenv("HF_TOKEN")
  if hf_token:
    model.push_to_hub(output_dir, token=hf_token)
    tokenizer.push_to_hub(output_dir, token=hf_token)
    print(f"Successfully pushed the fine-tuned model to the Hugging Face Hub under '{output_dir}'.")
  else:
    print("HF_TOKEN not set. Model saved locally only.")


if __name__ == "__main__":
  main()
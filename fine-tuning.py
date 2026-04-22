from datasets import load_dataset 
import re
import os
ds = load_dataset("summykai/chemistry-sft-ultra")
print([ds[1]])
from unsloth import is_bfloat16_supported 
from unsloth import FastLanguageModel
import torch
max_sequence_length = 1024
lora_rank = 32

SYSTEM = (
  "You are a chemistry expert, and you are helping a student to answer the question."
  " Show formulas and units in your answer, and explain the steps to get the answer when relevant."
)

model,tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-4-mini-reasoning-unsloth-bnb-4bit-Instruct",
    max_sequence_length=max_sequence_length,
    lora_rank=lora_rank,
    load_in_4bit=True,
    fast_tokenizer=True,)

def correctness_reward_func(prompts,completions,answer,**kwargs) ->list[float]:
  def _extract_number(text: str):
    if not text:
      return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text).replace(",", ""))
    return float(match.group(0)) if match else None

  def _has_chem_unit(text: str) -> bool:
    if not text:
      return False
    unit_keywords = ["mol", "g", "kg", "mg", "l", "ml", "m", "pa", "kpa", "atm", "k", "j", "kj"]
    lowered = str(text).lower()
    return any(unit in lowered for unit in unit_keywords)

  rewards = []
  for completion, ref_answer in zip(completions, answer):
    score = 0.0
    completion_text = str(completion) if completion is not None else ""
    ref_text = str(ref_answer) if ref_answer is not None else ""

    if completion_text.strip():
      score += 0.1

    lowered = completion_text.lower()
    if "step" in lowered or "therefore" in lowered or "so," in lowered:
      score += 0.2

    if _has_chem_unit(completion_text):
      score += 0.3

    pred_num = _extract_number(completion_text)
    gold_num = _extract_number(ref_text)
    if pred_num is not None and gold_num is not None:
      tolerance = 0.02 * max(1.0, abs(gold_num))
      if abs(pred_num - gold_num) <= tolerance:
        score += 0.4

    rewards.append(float(min(score, 1.0)))

  return rewards

  
def pick(row, names, default=""):
  for n in names:
    if n in row and row[n] is not None:
      return str(row[n]).strip()
  return default


def to_sft_text(row):
  # Try common schemas
  question = pick(row, ["question", "prompt", "instruction", "input", "problem"])
  answer = pick(row, ["answer", "response", "output", "completion", "solution"])
  prompt = f"{SYSTEM}\n\nProblem:\n{question}\n\n..."
  return {"prompt": prompt, "response": answer}
  
sft_ds = ds.map(to_sft_text)

from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
  use_vllm=True,
  learning_rate=5e-4,
  adam_beta1=0.8,
  adam_beta2=0.90,
  weight_decay=0.1,
  warmup_ratio=0.07,
  lr_scheduler_type="cosine",
  optim="paged_adamw_8bit",
  logging_steps=1,
  bf16=is_bfloat16_supported(),
  fp16=not is_bfloat16_supported(),
  per_device_train_batch_size=1,
  gradient_accumulation_steps=4,
  num_generations=6,
)   

trainer = GRPOTrainer(
  model=model,
  processing_class=tokenizer,
  reward_funcs=[correctness_reward_func],
  args=training_args,
  train_dataset=sft_ds,
)

trainer_stats = trainer.train()

model.save_pretrained("Chem-PHI-mini-4bit-lora")
tokenizer.save_pretrained("Chem-PHI-mini-4bit-lora")
hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_TOCKEN")
if not hf_token:
  raise ValueError("Set HF_TOKEN in your environment before pushing to the Hub.")

model.push_to_hub("Chem-PHI-mini-4bit-lora", token=hf_token)
tokenizer.push_to_hub("Chem-PHI-mini-4bit-lora", token=hf_token)

print("Successfully pushed the fine-tuned model to the Hugging Face Hub under the name 'Chem-PHI-mini-4bit-lora'.")
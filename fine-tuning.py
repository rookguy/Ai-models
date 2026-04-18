from datasets import load_dataset 
ds = load_dataset("summykai/chemistry-sft-ultra")

print([ds[1]])

from unsloth import is_bfloat16_supported 
import FastLanguageModel
import torch
max_sequence_length = 1024
lora_rank = 32

SYSTEM = (
  "You are a chemistry expert, and you are helping a student to answer the question."
  "show formula and units in your answer, and explain the steps to get the answer when relevant"
)

model,tockenizer = FastlanguageModel.from_pretained(
    model_name="unsloth/Phi-4-mini-reasoning-unsloth-bnb-4bit-Instruct",
    max_sequence_length=max_sequence_length,
    lora_rank=lora_rank,
    load_in_4bit=True,
    fast_tokenizer=True,)

def correctness_reward_func(prompts,completions,answer,**kwargs) ->list[float]:
  
  print(ds[0])

  
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
  adam_beta1=0.9,
  adam_beta2=0.95,
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
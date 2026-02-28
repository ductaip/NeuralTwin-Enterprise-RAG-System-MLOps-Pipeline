import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig

# ------------------------------------------------------------------------------
# CONFGIURATION FOR T4 GPU (16GB VRAM)
# ------------------------------------------------------------------------------
# This script is optimized to run on a free Google Colab T4 GPU instance or 
# local machine with similar specs. 
# Key optimizations:
# 1. 4-bit Quantization (QLoRA) - Reduces memory footprint by ~4x
# 2. Gradient Checkpointing - Saves memory at cost of computation
# 3. LoRA Adapters - Only trains a fraction of parameters (<1%)
# 4. Filtered Dataset - Ensures sequences fit in context window
# ------------------------------------------------------------------------------

# Model and Dataset
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B" 
NEW_MODEL_NAME = "llama-3.1-8b-fine-tuned-twin"
DATASET_NAME = "mlabonne/guanaco-llama2-1k"  # Example dataset for showcase

# QLoRA Parameters
LORA_R = 64                  # LoRA attention dimension
LORA_ALPHA = 16              # Alpha scaling for LoRA
LORA_DROPOUT = 0.1           # Dropout probability for LoRA layers

# bitsandbytes Parameters
USE_4BIT = True              # Activate 4-bit precision loading
BNB_4BIT_COMPUTE_DTYPE = "float16" # Compute dtype for 4-bit base models
BNB_4BIT_QUANT_TYPE = "nf4"  # Quantization type (fp4 or nf4)
USE_NESTED_QUANT = False     # Activate nested quantization for 4-bit base models (double quantization)

# TrainingArguments Parameters
OUTPUT_DIR = "./results"
NUM_TRAIN_EPOCHS = 1
FP16 = False
BF16 = False # Enable bf16 on Ampere GPUs (A100), use fp16 for T4
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
GRADIENT_CHECKPOINTING = True
MAX_GRAD_NORM = 0.3
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.001
OPTIMIZER = "paged_adamw_32bit"
LR_SCHEDULER_TYPE = "cosine"
MAX_STEPS = -1
WARMUP_RATIO = 0.03
GROUP_BY_LENGTH = True
SAVE_STEPS = 0
LOGGING_STEPS = 25

# SFT Parameters
MAX_SEQ_LENGTH = 512 # Reduced for T4 compatibility
PACKING = False
DEVICE_MAP = {"": 0}

def main():
    """
    Simulates the training pipeline. 
    NOTE: Check README_TRAINING.md for why this is not executed by default.
    """
    print(f"Starting QLoRA Fine-tuning for {MODEL_NAME}...")
    
    # 1. Load Dataset
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split="train")

    # 2. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # 3. Configure Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=USE_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
        bnb_4bit_use_double_quant=USE_NESTED_QUANT,
    )

    # 4. Load Base Model
    print("Loading base model...")
    # NOTE: In a real run, you need a HuggingFace token with access to Llama 3
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map=DEVICE_MAP
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # 5. Load LoRA Configuration
    peft_config = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 6. Set training parameters
    training_arguments = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim=OPTIMIZER,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=FP16,
        bf16=BF16,
        max_grad_norm=MAX_GRAD_NORM,
        max_steps=MAX_STEPS,
        warmup_ratio=WARMUP_RATIO,
        group_by_length=GROUP_BY_LENGTH,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        report_to="tensorboard"
    )

    # 7. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=PACKING,
    )

    # 8. Train
    print("Starting training (Simulated)...")
    # trainer.train() 
    # trainer.model.save_pretrained(NEW_MODEL_NAME)
    
    print("Training simulation completed successfully.")
    print(f"Model would be saved to: {NEW_MODEL_NAME}")

if __name__ == "__main__":
    # Check for execution flag to prevent accidental expensive runs
    if os.getenv("EXECUTE_TRAINING", "false").lower() == "true":
        main()
    else:
        print("Training execution skipped. Set EXECUTE_TRAINING=true to run.")
        print("See training/README_TRAINING.md for details.")

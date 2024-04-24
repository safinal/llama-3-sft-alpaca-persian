# To use DDP (which is generally recommended, see here for more info) you must launch the script with 
# python -m torch.distributed.launch script.py or accelerate launch script.py

import os

import torch
from accelerate import Accelerator, PartialState
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed, BitsAndBytesConfig

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

import wandb
wandb.login(key="7b872b88b3078c661b16dbbfc331e33174d35e04")


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = example['text']
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# def prepare_sample_text(example):
#     """Prepare the text from a sample of the dataset."""
#     text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
#     return text


def create_datasets(
        tokenizer, 
        dataset_name, 
        subset, 
        split, 
        streaming, 
        num_workers, 
        size_valid_set, 
        fraction_valid_set, 
        seq_length,
        seed
):
    dataset = load_dataset(
        dataset_name,
        data_dir=subset,
        split=split,
        num_proc=num_workers if not streaming else None,
        streaming=streaming,
    )
    if streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(size_valid_set)
        train_data = dataset.skip(size_valid_set)
        train_data = train_data.shuffle(seed=seed)
    else:
        dataset = dataset.train_test_split(test_size=fraction_valid_set, seed=seed)
        train_data = dataset["train"]
        # train_data = train_data.shuffle(seed=seed).select(range(1000)) # Only use 1000 samples for quick demo
        valid_data = dataset["test"]
        # valid_data = valid_data.shuffle(seed=seed).select(range(1000)) # Only use 1000 samples for quick demo
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")


    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

#     train_dataset = ConstantLengthDataset(
#         tokenizer,
#         train_data,
#         formatting_func=None,
#         infinite=True,
#         seq_length=seq_length,
#         chars_per_token=chars_per_token,
#     )
#     valid_dataset = ConstantLengthDataset(
#         tokenizer,
#         valid_data,
#         formatting_func=None,
#         infinite=False,
#         seq_length=seq_length,
#         chars_per_token=chars_per_token,
#     )
#     return train_dataset, valid_dataset
    return train_data, valid_data



def run_training(
        train_data, 
        val_data, 
        output_dir, 
        max_steps, 
        eval_freq, 
        save_freq, 
        log_freq, 
        batch_size, 
        learning_rate, 
        lr_scheduler_type, 
        num_warmup_steps, 
        gradient_accumulation_steps, 
        gradient_checkpointing, 
        fp16, 
        bf16,
        weight_decay,
        model_path
):
    print("Loading the model")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )

    train_data.start_iteration = 0

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=output_dir,
        dataloader_drop_last=True,
        eval_strategy="steps",
        max_steps=max_steps,
        eval_steps=eval_freq,
        save_steps=save_freq,
        logging_steps=log_freq,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=num_warmup_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        fp16=fp16,
        bf16=bf16,
        weight_decay=weight_decay,
        run_name="llama-3-8b-LoRA-alpaca-persian-finetuned",
        report_to="wandb",
        ddp_find_unused_parameters=False,
    )
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        attn_implementation="eager",
        device_map='auto' # device_map={"": Accelerator().process_index} {"": PartialState().process_index}
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
        packing=True,
        dataset_text_field='text'
    )

    print_trainable_parameters(trainer.model)

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    trainer.model.save_pretrained(os.path.join(output_dir, "final_checkpoint/"))


def main(
        dataset_name, 
        subset, 
        split, 
        streaming, 
        num_workers, 
        size_valid_set, 
        fraction_valid_set, 
        seq_length, 
        seed,
        output_dir, 
        max_steps, 
        eval_freq, 
        save_freq, 
        log_freq, 
        batch_size, 
        learning_rate, 
        lr_scheduler_type, 
        num_warmup_steps, 
        gradient_accumulation_steps, 
        gradient_checkpointing, 
        fp16, 
        bf16,
        weight_decay,
        model_path
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_data, val_data = create_datasets(
        tokenizer, 
        dataset_name, 
        subset, 
        split, 
        streaming, 
        num_workers, 
        size_valid_set, 
        fraction_valid_set, 
        seq_length, 
        seed
    )
    run_training(
        train_data, 
        val_data, 
        output_dir, 
        max_steps, 
        eval_freq, 
        save_freq, 
        log_freq, 
        batch_size, 
        learning_rate, 
        lr_scheduler_type, 
        num_warmup_steps, 
        gradient_accumulation_steps, 
        gradient_checkpointing, 
        fp16, 
        bf16,
        weight_decay,
        model_path
    )


DATASET_NAME = "sinarashidi/alpaca-persian"
SUBSET = None # Subset Folder of the dataset
SPLIT = "train"
STREAMING = False
NUM_WORKERS = None # Multithreadedly downloading the dataset
SIZE_VALID_SET = None
FRACTION_VALID_SET = 0.1
SEQ_LENGTH = 2048
SEED = 42
OUTPUT_DIR = "./llama-3-8b-LoRA-alpaca-persian-finetuned"
MAX_STEPS = -1
EVAL_FREQ = None
SAVE_FREQ = 10
LOG_FREQ = 10
BATCH_SIZE = 1
LEARNING_RATE = 0.0001
LR_SCHEDULER_TYPE = "linear"
NUM_WARMUP_STEPS = 0
GRADIENT_ACCUMULATION_STEPS = 1
GRADIENT_CHECKPOINTING = True
FP16 = True
BF16 = False
WEIGHT_DECAY = 0
MODEL_PATH = "meta-llama/Meta-Llama-3-8B"

set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.set_verbosity_error()
main(
    dataset_name=DATASET_NAME, 
    subset=SUBSET, 
    split=SPLIT, 
    streaming=STREAMING, 
    num_workers=NUM_WORKERS, 
    size_valid_set=SIZE_VALID_SET, 
    fraction_valid_set=FRACTION_VALID_SET, 
    seq_length=SEQ_LENGTH, 
    seed=SEED,
    output_dir=OUTPUT_DIR, 
    max_steps=MAX_STEPS, 
    eval_freq=EVAL_FREQ, 
    save_freq=SAVE_FREQ, 
    log_freq=LOG_FREQ, 
    batch_size=BATCH_SIZE, 
    learning_rate=LEARNING_RATE, 
    lr_scheduler_type=LR_SCHEDULER_TYPE, 
    num_warmup_steps=NUM_WARMUP_STEPS, 
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, 
    gradient_checkpointing=GRADIENT_CHECKPOINTING, 
    fp16=FP16, 
    bf16=BF16,
    weight_decay=WEIGHT_DECAY,
    model_path=MODEL_PATH
)

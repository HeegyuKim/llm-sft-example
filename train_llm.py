from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer
import fire


def main(
    model_name_or_path: str = "meta-llama/Llama-3.1-8B",
    tokenizer_name_or_path: str = None,
    dataset_name: str = "trl-lib/chatbot_arena_completions",
    train_limit: int = 10000,
    push_to_hub: bool = False,
    push_to_hub_model_id: str = None,
    per_device_train_batch_size: int = 1,
    total_batch_size: int = 32,
):


    dataset = load_dataset(dataset_name)
    if train_limit:
        dataset["train"] = dataset["train"].select(range(train_limit))

    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = model_name_or_path
    else:
        tokenizer_name_or_path = tokenizer_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_config = SFTConfig(
        max_seq_length=1024,
        output_dir="./training_output",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=total_batch_size // per_device_train_batch_size,
        save_strategy="epoch",
        num_train_epochs=1,
        bf16=True,
        logging_steps=1,
        gradient_checkpointing=False,
        push_to_hub=push_to_hub,
        hub_model_id=push_to_hub_model_id,
    )

    trainer = SFTTrainer(
        model_name_or_path,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=sft_config,
    )
    trainer.train()

if __name__ == "__main__":
    fire.Fire(main)
# https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/discussions/31

import torch
import fire
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration
from peft import LoraConfig
from trl import (
    SFTConfig,
    SFTTrainer
)

def main(
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct",
    dataset_name = "ayoubkirouane/llava-instruct-small", # mini (500) dataset
    train_limit: int = None,
    push_to_hub: bool = False,
    push_to_hub_model_id: str = None,
    per_device_train_batch_size: int = 1,
    total_batch_size: int = 32,
    gradient_checkpointing: bool = True,
    peft: bool = False,
    bits: int = None,
):
    model_kwargs = {}
    if bits:
        from transformers import BitsAndBytesConfig
        if bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif bits == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        peft = True
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(model_id, **model_kwargs)

    def collate_fn(examples):
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        images = [example["images"] for example in examples]

        if isinstance(model, LlavaForConditionalGeneration):
            images = [image[0] for image in images]

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  #
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch

    dataset = load_dataset(dataset_name)
    if train_limit:
        dataset["train"] = dataset["train"].select(range(train_limit))

    sft_config = SFTConfig(
        max_seq_length=512,
        output_dir="./training_output",
        save_strategy="epoch",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=total_batch_size // per_device_train_batch_size,
        num_train_epochs=2,
        remove_unused_columns=False,
        logging_steps=1,
        bf16=True,
        push_to_hub=push_to_hub,
        hub_model_id=push_to_hub_model_id,
        gradient_checkpointing=gradient_checkpointing,
        dataset_kwargs={"skip_prepare_dataset":True}
    )

    if peft:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        data_collator=collate_fn,
        peft_config=peft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
        tokenizer=processor.tokenizer,
    )

    trainer.train()
    
if __name__ == "__main__":
    fire.Fire(main)
# llm-sft-example

## Setup
```
pip install torch # for your environment
pip install flash-attn --no-build-isolation
# install requirements
pip install -r requirements.txt

# Login huggingface
huggingface-cli login
```

## Train
### LLM SFT
```bash
# 8B
python train_llm.py \
    --model_name_or_path meta-llama/Llama-3.1-8B \
    --tokenizer_name_or_path meta-llama/Llama-3.1-8B-Instruct

# 3B
python train_llm.py \
    --model_name_or_path meta-llama/Llama-3.2-3B \
    --tokenizer_name_or_path meta-llama/Llama-3.2-3B-Instruct

# with accelerate / Deepspeed CPU offload (optimizer)
accelerate launch \
    --config_file single-ds0.yaml \
    train_llm.py \
    --model_name_or_path meta-llama/Llama-3.2-3B \
    --tokenizer_name_or_path meta-llama/Llama-3.2-3B-Instruct
```


### VLM SFT
```bash
# Phi-3 vision 4B
python train_vlm.py \
    --model_name_or_path microsoft/Phi-3.5-vision-instruct

# Llama 3.2 Vision 11B
python train_vlm.py

```
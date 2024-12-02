---
base_model: /root/lanyun-tmp/huggingface/Qwen/Qwen2.5-7B-Instruct
library_name: peft
license: other
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: train_2024-10-18-02-56-15
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# train_2024-10-18-02-56-15

This model is a fine-tuned version of [/root/lanyun-tmp/huggingface/Qwen/Qwen2.5-7B-Instruct](https://huggingface.co//root/lanyun-tmp/huggingface/Qwen/Qwen2.5-7B-Instruct) on the huatuo_qa_train dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 4
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 10
- total_train_batch_size: 40
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- num_epochs: 8.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.13.2
- Transformers 4.38.1
- Pytorch 2.4.1+cu121
- Datasets 3.0.1
- Tokenizers 0.15.2
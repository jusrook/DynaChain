from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
import torch
from unsloth.chat_templates import train_on_responses_only
from unsloth import FastLanguageModel
from datasets import Dataset
import argparse
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers.trainer_callback import EarlyStoppingCallback
import os
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_args():
    parse = argparse.ArgumentParser(description="训练因果语言模型")
    parse.add_argument('--model_root', type=str, required=True, help='预训练模型路径')
    parse.add_argument('--train_data_path', type=str, required=True, help='训练数据路径')
    parse.add_argument('--val_data_path', type=str, required=True, help='验证数据路径')
    parse.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parse.add_argument('--instruction_part', type=str, default="<|start_header_id|>user<|end_header_id|>\n\n")
    parse.add_argument('--response_part', type=str, default="<|start_header_id|>assistant<|end_header_id|>\n\n")
    parse.add_argument('--batch_size', type=int, default=128)
    parse.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parse.add_argument('--num_train_epochs', type=int, default=2)
    parse.add_argument('--learning_rate', type=float, default=5e-5)
    parse.add_argument('--logging_steps', type=int, default=20)
    parse.add_argument('--eval_steps', type=int, default=100)
    parse.add_argument('--save_steps', type=int, default=200)
    parse.add_argument('--early_stopping_patience', type=int, default=3)
    parse.add_argument('--use_lora', type=lambda x: x.lower() == 'true', default=True)
    parse.add_argument('--lora_r', type=int, default=16)
    parse.add_argument('--lora_alpha', type=int, default=32)
    parse.add_argument('--lora_dropout', type=float, default=0.1)
    parse.add_argument('--max_seq_length', type=int, default=4096, help='最大序列长度')
    
    return parse.parse_args()

def load_model_and_tokenizer(model_root, args):

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_root,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def setup_lora(model, args):
    if args.use_lora:
        # FastLanguageModel 一键加 LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing=True,   # 省显存
            random_state=42,
        )
    return model

def prepare_datasets(train_path, val_path, tokenizer, max_length):
    """准备数据集"""
    train_ds = Dataset.from_csv(train_path)
    eval_ds = Dataset.from_csv(val_path)
    
    logger.info(f"训练集大小: {len(train_ds)}")
    logger.info(f"验证集大小: {len(eval_ds)}")
    
    def apply_template(ds):
        messages = [
            {"role": "user", "content": ds["prompt"]},
            {"role": "assistant", "content": ds["response"]}
        ]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": input_text}
    
    def tokenize_function(ds):
        tokenized = tokenizer(
            ds["text"], 
            truncation=True, 
            padding=False,
            max_length=max_length
        )
        return tokenized
    
    # 应用模板和分词
    train_template_ds = train_ds.map(apply_template)
    eval_template_ds = eval_ds.map(apply_template)
    
    train_tk_ds = train_template_ds.map(tokenize_function, remove_columns=train_template_ds.column_names)
    eval_tk_ds = eval_template_ds.map(tokenize_function, remove_columns=eval_template_ds.column_names)
    
    return train_tk_ds, eval_tk_ds

def main():
    args = setup_args()
    instruction_part = args.instruction_part.replace('\\n', '\n')
    response_part = args.response_part.replace('\\n', '\n')
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    # import debugpy  
    # debugpy.listen(("0.0.0.0", 5678))  # 监听所有 IP，端口可改 
    # print("  Waiting for debugger attach on port 5678...") 
    # debugpy.wait_for_client()  # 等待调试器连接 
    # debugpy.breakpoint()
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(args.model_root, args)
    
    # 配置LoRA
    model = setup_lora(model, args)
    
    # 准备数据集
    train_dataset, eval_dataset = prepare_datasets(
        args.train_data_path, 
        args.val_data_path, 
        tokenizer, 
        args.max_seq_length
    )
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        eval_strategy="steps",
        eval_accumulation_steps=1,
        save_strategy="steps",
        save_total_limit=3,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="tensorboard",
        remove_unused_columns=False,
        average_tokens_across_devices=False
    )
    
    # 设置回调
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        ))
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        tokenizer=tokenizer,
        callbacks=callbacks
    )
    
    # 应用response-only训练
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part=response_part,
    )
    
    # 开始训练
    train_result = trainer.train()
    
    # 保存结果
    trainer.save_model()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # 最终评估
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    logger.info(f"训练完成！模型保存在: {args.output_dir}")

if __name__ == "__main__":
    main()
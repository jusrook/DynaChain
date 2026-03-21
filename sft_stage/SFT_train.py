from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import torch
# from unsloth.chat_templates import train_on_responses_only
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
    """加载模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(model_root)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 设备映射策略
    # device_map = "auto" if not torch.distributed.is_initialized() else None
    device_map = None
    model = AutoModelForCausalLM.from_pretrained(
        model_root,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2"
    )
    
    return model, tokenizer

def setup_lora(model, args):
    """配置LoRA"""
    if args.use_lora:
        # if args.load_in_4bit:
        #     model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
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

def _longest_common_substring(arr):
    n = len(arr)
    s = arr[0]
    l = len(s)
    res = ""
    for i in range(l):
        for j in range(i + 1, l + 1):
            stem = s[i:j]
            k = 1
            for k in range(1, n):
                if stem not in arr[k]:
                    break
            if (k + 1 == n and len(res) < len(stem)):
                res = stem
    return res
pass

def _find_common_token_ids(component, tokenizer):
    """
    \n### User:\n\n
    \n\n### User:\n\n
    etc
    we need to find the middle most repeatted part.
    Tokenizers can tokenize newlines or spaces as 1 token!
    """
    right_text = ""
    if   component.endswith (" "): right_text = " "
    elif component.endswith("\n"): right_text = "\n"
    left_text = ""
    if   component.startswith (" "): left_text = " "
    elif component.startswith("\n"): left_text = "\n"
    stripped = component.strip()

    # Add current pieces and also newlines
    all_input_ids = []
    for left in range(3):
        for right in range(3):
            x = left*left_text + stripped + right*right_text
            x = tokenizer(x, add_special_tokens = False).input_ids
            all_input_ids.append(x)

            x = left*"\n" + stripped + right*"\n"
            x = tokenizer(x, add_special_tokens = False).input_ids
            all_input_ids.append(x)
        pass
    pass
    substring = _longest_common_substring([str(x + [0]) for x in all_input_ids])
    substring = substring.split(", ")[:-1]
    substring = [part for part in substring if part]
    substring = [int(x) for x in substring]

    # Also get rest of tokenized string
    original = tokenizer(component, add_special_tokens = False).input_ids
    # Get optional left and right
    for j in range(len(original)):
        if original[j : j + len(substring)] == substring: break
    optional_left  = original[:j]
    optional_right = original[j+len(substring):]
    return substring, optional_left, optional_right
pass


def train_on_responses_only(
    trainer,
    instruction_part = None,
    response_part    = None,
):
    """
    Trains only on responses and not on the instruction by masking out
    the labels with -100 for the instruction part.
    """
    tokenizer = trainer.tokenizer
    
    if  not hasattr(tokenizer, "_unsloth_input_part") or \
        not hasattr(tokenizer, "_unsloth_output_part"):
        
        if instruction_part is None or response_part is None:
            raise ValueError("Unsloth: instruction_part and response_part must be given!")
        pass
    elif (instruction_part is not None or response_part is not None) and \
        (hasattr(tokenizer, "_unsloth_input_part") or hasattr(tokenizer, "_unsloth_output_part")):

        raise ValueError("Unsloth: Your tokenizer already has instruction and response parts set - do not give custom ones!")
    else:
        instruction_part = tokenizer._unsloth_input_part
        response_part    = tokenizer._unsloth_output_part
    pass

    # Get most common tokens since tokenizers can tokenize stuff differently!
    Q_must, Q_left, Q_right = _find_common_token_ids(instruction_part, tokenizer)
    A_must, A_left, A_right = _find_common_token_ids(response_part,    tokenizer)

    # Store some temporary stuff
    A_first = A_must[0]
    len_A_must = len(A_must)
    A_left_reversed = A_left[::-1]
    A_right_forward = A_right

    Q_first = Q_must[0]
    len_Q_must = len(Q_must)
    Q_left_reversed = Q_left[::-1]
    Q_right_forward = Q_right

    def _train_on_responses_only(examples):
        input_ids_ = examples["input_ids"]
        all_labels = []

        for input_ids in input_ids_:
            n = len(input_ids)
            labels = [-100] * n
            n_minus_1 = n - 1
            j = 0
            while j < n:
                # Find <assistant>
                if (input_ids[j] == A_first) and \
                    (input_ids[j : (k := j + len_A_must)] == A_must):

                    # Now backtrack to get previous optional tokens
                    for optional_left in A_left_reversed:
                        if j < 1: break
                        if optional_left == input_ids[j-1]: j -= 1
                        else: break
                    pass
                    # And forwards look as well
                    for optional_right in A_right_forward:
                        if k >= n_minus_1: break
                        if optional_right == input_ids[k+1]: k += 1
                        else: break
                    pass
                    # assistant_j = j
                    assistant_k = k

                    j = assistant_k
                    # Given <assistant>, now find next user
                    while j < n:
                        # Find <user>
                        # Also accept last final item if assistant is the last turn
                        if (j == n_minus_1) or \
                            ((input_ids[j] == Q_first) and \
                             (input_ids[j : (k := j + len_Q_must)] == Q_must)):

                            # Now backtrack to get previous optional tokens
                            for optional_left in Q_left_reversed:
                                if j < 1: break
                                if optional_left == input_ids[j-1]: j -= 1
                                else: break
                            pass
                            # And forwards look as well
                            for optional_right in Q_right_forward:
                                if k >= n_minus_1: break
                                if optional_right == input_ids[k+1]: k += 1
                                else: break
                            pass
                            user_j = j
                            # Account for last item
                            if user_j != n_minus_1:
                                # user_k = k
                                # j = user_k
                                j = k
                            else:
                                user_j = n
                                k = n
                            pass
                            # Now copy input_ids to labels
                            labels[assistant_k : user_j] = input_ids[assistant_k : user_j]
                            # print(assistant_j, assistant_k, user_j, user_k)
                            break
                        pass
                        j += 1
                    pass
                pass
                j += 1
            pass
            all_labels.append(labels)
        pass
        return { "labels" : all_labels }
    pass

    if hasattr(trainer, "train_dataset") and trainer.train_dataset is not None:
        trainer.train_dataset = trainer.train_dataset.map(_train_on_responses_only, batched = True)
    if hasattr(trainer, "eval_dataset") and trainer.eval_dataset is not None:
        trainer.eval_dataset = trainer.eval_dataset.map(_train_on_responses_only, batched = True)
    return trainer
pass

def main():
    args = setup_args()
    instruction_part = args.instruction_part.replace('\\n', '\n')
    response_part = args.response_part.replace('\\n', '\n')
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print("main中开始") 
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
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        tokenizer=tokenizer,
        callbacks=callbacks
    )
    mx = 0
    for i in range(len(train_dataset)):
        l = len(train_dataset[i]['input_ids'])
        if l > mx:
            mx = l
    print("max len:", mx)
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
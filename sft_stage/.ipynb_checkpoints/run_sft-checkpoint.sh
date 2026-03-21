# 1. 路径与数据
MODEL_ROOT="/root/autodl-tmp/AI-ModelScope/Mistral-7B-Instruct-v0___2"   # 预训练模型路径
TRAIN_DATA="/root/autodl-fs/data/safety_aware_train_sft_data.csv"                     # 训练集
VAL_DATA="/root/autodl-fs/data/safety_aware_val_sft_data.csv"                         # 验证集
OUTPUT_DIR="/root/autodl-tmp/outputs/12_16_Mistral-7B-Instruct_sft_lora"  # 添加时间戳避免覆盖

# 2. 训练超参
BATCH_SIZE=16                  # 单卡 batch_size
GRAD_ACCUM=4                    # 梯度累积
MAX_SEQ_LEN=4096
EPOCHS=3
LR=5e-5
SAVE_STEPS=200
EVAL_STEPS=100
LOG_STEPS=20
EARLY_STOP=3

# 3. LoRA 设置
LORA_R=16
LORA_ALPHA=32
LORA_DROP=0.1

# 4. 其它开关
USE_LORA=true                   # 使用 LoRA

# 5. GPU设置（新增）
CUDA_VISIBLE_DEVICES="0"  # 指定使用的GPU
NPROC=1
# ===================================================

# 设置GPU
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 保存训练配置（新增）
cat > "$OUTPUT_DIR/train_config.txt" << EOF
训练启动时间: $(date)
模型路径: $MODEL_ROOT
训练数据: $TRAIN_DATA
验证数据: $VAL_DATA
输出目录: $OUTPUT_DIR
批次大小: $BATCH_SIZE
梯度累积: $GRAD_ACCUM
最大序列长度: $MAX_SEQ_LEN
训练轮数: $EPOCHS
学习率: $LR
保存步数: $SAVE_STEPS
评估步数: $EVAL_STEPS
日志步数: $LOG_STEPS
早停耐心值: $EARLY_STOP
LoRA秩: $LORA_R
LoRA Alpha: $LORA_ALPHA
LoRA Dropout: $LORA_DROP
使用LoRA: $USE_LORA
使用的GPU: $CUDA_VISIBLE_DEVICES
EOF

# 打印参数
echo "=========================================="
cat "$OUTPUT_DIR/train_config.txt"
echo "=========================================="

# 启动训练
echo "开始训练..."
torchrun --nproc_per_node=${NPROC:-1} /root/autodl-fs/sft_stage/SFT_train.py \
  --model_root "$MODEL_ROOT" \
  --train_data_path "$TRAIN_DATA" \
  --val_data_path "$VAL_DATA" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --max_seq_length $MAX_SEQ_LEN \
  --num_train_epochs $EPOCHS \
  --learning_rate $LR \
  --save_steps $SAVE_STEPS \
  --eval_steps $EVAL_STEPS \
  --logging_steps $LOG_STEPS \
  --early_stopping_patience $EARLY_STOP \
  --lora_r $LORA_R \
  --lora_alpha $LORA_ALPHA \
  --lora_dropout $LORA_DROP \
  --use_lora $USE_LORA \
  --instruction_part "[INST] " \
  --response_part " [/INST]" \

# 训练完成处理
if [ $? -eq 0 ]; then
    echo "训练成功完成！模型保存在：$OUTPUT_DIR"
else
    echo "训练失败，请检查日志：$OUTPUT_DIR/training.log"
    exit 1
fi
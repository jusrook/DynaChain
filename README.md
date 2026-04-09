本仓库支持监督微调（SFT）和强化学习（RL）两个阶段的训练。

## 准备工作

1. **数据配置**：在运行脚本之前，请确保在相应的配置文件中填写正确的数据路径。
2. **模型路径**：检查并修改脚本中的模型路径，确保指向正确的预训练模型或检查点。
3. **下载奖励模型**：RL 阶段需要用到奖励模型，请将对应的模型文件下载至 `verl/custom_reward/` 目录下。

## 运行 SFT

执行以下命令启动 SFT 训练：

```bash
cd sft_stage
sh run_sft.sh
```

## 运行 RL

执行以下命令启动 RL 训练：

```bash
cd verl
sh recipe/dapo/run_dapo_recipe.sh
```

注意：请根据实际需求修改脚本内的数据路径、模型路径及超参数。

奖励所用prompt路径：https://github.com/jusrook/DynaChain/blob/main/verl/custom_reward/custom_math_reward.py

测试所用prompt路径：https://github.com/jusrook/Eval-JB/blob/main/eval_client.py

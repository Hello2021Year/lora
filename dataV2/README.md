# dataV2 · Qwen3.8-27B 双机 64K LoRA

本目录内的 [原始数据包](jev_datasets_v2.zip) 与训练机上的目标文件同名。上游版本的所有最终 assistant 答案均为小写 `yes` 或 `no`，system 提示也要求小写。SHA-256 为 `9df7eab8c1188bf701859250c55ebb1826cdbd414771528025c75af69f3f964a`。[Hugging Face 固定版本](https://huggingface.co/datasets/sunpengliang/test/blob/38d3e26eb35c2dbd65618c547982abfb693f554b/jev_datasets_v2.zip)

| 集合 | 原始 yes / no | 64K 保留 yes / no | 保留总数 |
|---|---:|---:|---:|
| train | 896 / 648 | 856 / 603 | 1,459 |
| validation | 112 / 81 | 107 / 80 | 187 |
| test | 112 / 81 | 84 / 65 | 149 |

上表按固定模型 tokenizer 和 compact 工具目录实测。预处理保留原有切分，按带模板的估算长度额外预留 32 token，过滤超限整条样本。运行时的完整统计在 `prepared/dataV2-compact64k/audit.json`。

## 1. 检查并准备数据

两台机器使用同一提交和相同的服务器路径。模型默认在 `/mnt_upfs/models/qwen3/Qwen3.8-27B`，Conda 环境在 `/mnt_upfs/conda/swift-qwen38`。以下命令都从仓库根目录运行。

```bash
# 两台各执行一次
DATA_VERSION=v2 bash run_server.sh check

# 在共享文件系统上执行一次；prepare 会核对服务器 ZIP 的 SHA-256
DATA_VERSION=v2 bash run_server.sh prepare
```

默认原包位置为 `/mnt_upfs/models/qwen3/test/jev_datasets_v2.zip`。使用仓库内的同一份包时，可给 check 和 prepare 指定 `RAW_DATA_DIR="$PWD/dataV2"`。prepare 会创建新目录；重新准备时，请另选输出目录或先将旧目录移开。若两台使用独立磁盘，将同一份准备好的目录复制到两端。

## 2. 先在 16 张卡上做长样本压测

两台机器分别启动相应的命令。将 `MASTER_ADDR` 替换为节点 0 的训练网 IP，两端使用同一端口。

```bash
# 节点 0
export MASTER_ADDR=10.0.0.10 MASTER_PORT=29561
bash run_16gpu.sh 0 smoke 64k
```

```bash
# 节点 1
export MASTER_ADDR=10.0.0.10 MASTER_PORT=29561
bash run_16gpu.sh 1 smoke 64k
```

smoke 的训练文件只取最长的 16 条，实测估长为 63,042–65,501 token。16 卡每卡 1 条，全局 batch 16，每次更新都覆盖这组长样本，共更新 3 步。第 3 步用最长的 16 条验证样本算验证 loss，并保存 smoke checkpoint。检查两端正常结束、loss 有限，逐卡比较日志中的 peak_allocated 与 peak_reserved，并结合 nvidia-smi 查看整卡余量。GPU 上 64K 是否能容纳以这一步为准。

## 3. 从基座启动正式训练

```bash
# 两端保留上面的 MASTER_ADDR 和 MASTER_PORT，各执行对应的一行
# 节点 0
bash run_16gpu.sh 0 train 64k
# 节点 1
bash run_16gpu.sh 1 train 64k
```

| 项目 | 配置 |
|---|---|
| 环境与序列 | MS-Swift，模型类型 qwen3_5，聊天模板 qwen3_8，max_length 65,536 |
| 训练 | BF16 LoRA SFT，ZeRO-3，FlashAttention 2，梯度检查点，use_logits_to_keep，不做 packing |
| LoRA | rank 8、alpha 16、dropout 0.05，学习率 5e-5，warmup 比例 0.05 |
| 更新 | 每卡 1 条，16 卡，梯度累积 1，全局 batch 16，正式训练 1 epoch |
| 权重位置 | 16 个 self_attn.o_proj、48 个 linear_attn.out_proj、64 个 mlp.down_proj |

每次启动时，回调核验实际 PEFT 模型中的插入位置及可训练参数数目。默认 rank 8 应命中 128 个模块、17,301,504 个参数。节点 0 打印完整模块列表。正式训练从基座初始化新的适配器，目录为 `outputs/dataV2-compact64k-16gpu-train/`。

## 4. 严格小写生成评估

在一台机器上运行。将 CKPT 设为训练日志里的实际 checkpoint 路径。对基座和 LoRA 使用同一份 validation 和生成参数，查看 format_pass、accuracy、macro_f1 和两类 recall。评分按原字符串匹配，大写、附加标点、空格或解释均计为格式错误。

```bash
export DATA_VERSION=v2 CUDA_VISIBLE_DEVICES=0,1
CKPT=outputs/dataV2-compact64k-16gpu-train/v0-实际时间戳/last
bash run_server.sh eval validation base-v2-64k
bash run_server.sh eval validation lora-v2-64k "$CKPT"
# 选定配置后跑最终测试
bash run_server.sh eval test lora-v2-64k-test "$CKPT"
```

原有 V1 启动方式沿用大写标签和 all-linear。评分器从 holdout 同目录的 audit.json 读取版本；对于单独复制出来且没有 audit 的 V2 JSONL，直接调用 score_eval.py 时传 `--data-version v2`。

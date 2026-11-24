# Qwen3.8-27B 工具选择判定微调

**dataV2 双机 64K** 使用两台各 8 张 A800 80G、严格小写 `yes/no` 和全 64 个语言 block 的 128 个输出投影。服务器从 `/mnt_upfs/models/qwen3/test/jev_datasets_v2.zip` 读取数据。准备、长样本压测、训练和验证命令见 [dataV2 运行说明](dataV2/README.md)。

## V1 实验

首轮用 **单机 8 × A800 80GB，BF16 LoRA SFT，MS-Swift 4.6.1**。输入是历史状态和一个候选工具，模型输出严格等于 `YES` 或 `NO`。另一台机器可以独立跑输入方案对照或部署离线推理。

## 服务器路径

| 内容 | 默认位置 |
| --- | --- |
| 基座模型 | `/mnt_upfs/models/qwen3/Qwen3.8-27B` |
| 数据目录 | `/mnt_upfs/models/qwen3/test` |
| 原始 ZIP | `/mnt_upfs/models/qwen3/test/jev_datasets_v1.zip` |
| 原始测试集 | `/mnt_upfs/models/qwen3/test/test.jsonl` |
| Conda 环境前缀 | `/mnt_upfs/conda/swift-qwen38` |

服务器入口 `run_server.sh` 使用已有 Conda 环境的 Python 和 Swift，读取本地 ZIP 与模型 tokenizer。已激活目标环境时直接使用当前环境，否则通过 `conda run --prefix` 执行。Conda 程序位置由当前 shell 查找，也可以通过 `CONDA_EXE` 指定。

## 在训练服务器执行

```bash
git clone https://github.com/Hello2021Year/lora.git
cd lora

bash run_server.sh check
bash run_server.sh prepare
bash run_server.sh smoke
bash run_server.sh train
```

`check` 检查模型配置、ZIP、Conda 中的依赖与可见 GPU。`prepare` 校验数据 ZIP 的 SHA-256，从 ZIP 读取原始 train、validation、test 切分，并用本地 tokenizer 准备数据。产物写入仓库下的 `prepared/compact48k/`，adapter 写入 `outputs/`，推理结果写入 `results/`。模型和原始数据目录可以只读挂载。

如果环境检查发现缺少依赖，可在已匹配服务器驱动的 CUDA PyTorch 环境里参考下面的组合。FlashAttention 2 和 causal-conv1d 使用与当前 Python、PyTorch 和 CUDA 一致的轮子或本地构建。长上下文 smoke 时检查 Gated DeltaNet 快速路径是否可用。[MS-Swift 模型支持](https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html) · [FLA 安装](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md) · [FlashAttention 安装](https://github.com/Dao-AILab/flash-attention#installation-and-features)

```bash
conda activate /mnt_upfs/conda/swift-qwen38
python3 -m pip install 'ms-swift==4.6.1' 'transformers==5.14.0' \
  'qwen-vl-utils>=0.0.14' decord deepspeed 'flash-linear-attention[cuda]'
python3 -m pip install --no-build-isolation flash-attn causal-conv1d
```

也可以用环境变量覆盖路径：

| 变量 | 用途 |
| --- | --- |
| `MODEL` | 基座模型目录 |
| `RAW_DATA_DIR` | 存放原始 ZIP 的目录 |
| `TRAIN_CONDA_PREFIX` | Conda 环境前缀 |
| `CONDA_EXE` | Conda 可执行文件的绝对路径 |
| `PREPARED_DATA_DIR` | 预处理数据目录 |
| `OUTPUT_DIR` | 训练输出目录 |

## 数据评估

ZIP 内的每行都符合 MS-Swift 的外层 `messages` 格式。外层 `user.content` 是一个 JSON 字符串，包含工具目录、历史消息、候选工具、控制动作和问题。训练时保留外层 system 指令，历史消息仍然放在该 JSON 内。数据版本固定至 `219df35b1b72694c03556047b7337d6a1a0474e9`。[Hugging Face 数据文件](https://huggingface.co/datasets/sunpengliang/test/tree/219df35b1b72694c03556047b7337d6a1a0474e9) · [MS-Swift 数据格式](https://swift.readthedocs.io/en/v4.6/Customization/Custom-dataset.html)

| 集合 | 原始样本 | YES / NO | 独立历史状态 | 不同起始用户请求 | 精简版 49,152 token 内 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 训练 | 1,544 | 896 / 648 | 839 | 125 | 1,321 / 1,544（85.6%） |
| 验证 | 193 | 112 / 81 | 104 | 13 | 172 / 193（89.1%） |
| 测试 | 193 | 112 / 81 | 108 | 12 | 138 / 193（71.5%） |

- 按外层用户输入、完整历史状态和起始用户请求分别做了精确哈希检查，三个集合之间均为零重叠。测试集的 193 条判定只来自 12 个不同起始请求，细小的分数差异要按请求成组查看。
- 训练集原始输入的长度中位数约 **44.2K**，P95 约 **90.0K**，最大约 **127.5K**。每条样本都带同一份 39 工具目录。精简版将工具目录项缩为 `kind/name/description`，候选工具的完整定义、控制动作和所有历史均保留。训练集精简后的中位数约 **21.4K**，P95 约 **67.2K**。
- 原始测试集全部猜 YES 的准确率为 **58.0%**，按训练集的候选工具名称选多数标签可达 **80.3%**。精简版 48K 测试子集上，按同样规则从实际保留的训练数据拟合，可达 **80.4% Accuracy / 80.0% Macro F1**。优化重点是工具名相同、状态不同且标签相反的样本。
- 精简版是首轮实验输入。线上复用 `prepare_data.py` 的 `compact_payload()`，并按线上收到的真实输入做回归。同一模型用全量目录和精简目录比较时，使用相同的留出样本。

## 训练参数

准备脚本校验 ZIP 的 SHA-256，估算带 ChatML 模板的 token 长度，额外留出 32 token，并完整筛掉超限样本。生成的 `audit.json` 记录切分、标签、覆盖率和长度分布。Smoke 使用 24 条训练记录，包含最长的 8 条，单机 8 卡执行 3 步并保存 adapter。进入正式训练前检查 loss 和 grad norm 有限、峰值显存有余量、checkpoint 中存在 `adapter_config.json` 与 adapter 权重。

正式训练为 1 epoch，LoRA rank 8、alpha 16、LR `5e-5`，每卡 microbatch 1、梯度累积 2、全局 batch 16，ZeRO-3 分片，训练范围为语言侧线性层。只监督最终标签和结束位置，忽略空思考块。启用 gradient checkpointing 和 `logits_to_keep`，避免为每个输入 token 生成全词表 logits。checkpoint 位于 `outputs/compact48k-train/v*/last` 所指目录，运行路径以日志为准。[MS-Swift 参数](https://swift.readthedocs.io/en/v4.6/Instruction/Command-line-parameters.html)

如果 48K smoke 显存不足，先确认 FlashAttention 和 Gated DeltaNet 快速路径已经启用，再使用下面的 32K 配置。32K 精简版训练、验证、测试覆盖分别是 1,036、140、118 条。

```bash
export PREPARED_DATA_DIR=prepared/compact32k
bash run_server.sh prepare --max-length 32768
bash run_server.sh smoke
bash run_server.sh train
unset PREPARED_DATA_DIR
```

## 验收

用相同的 prompt、模型版本和解码设置评估原模型与 LoRA。先在验证集选方案，锁定后跑测试集。评估脚本会核对预测覆盖及标签，`YES.`、小写、空串或附带解释都会计为格式错误。结果包括严格准确率、Macro F1、YES/NO 召回率及两个规则基线。

```bash
# 填入已完成 SFT 的实际 checkpoint 或该次运行的 last 链接。
CKPT=outputs/compact48k-train/v0-实际时间戳/last
bash run_server.sh eval validation base-val
bash run_server.sh eval validation lora-val "$CKPT"
bash run_server.sh eval test base-test
bash run_server.sh eval test lora-test "$CKPT"
```

建议验收线为 **格式通过率 100%**，Macro F1 超过原模型和候选名规则，逐项检查 YES/NO 召回和关键错误。生产接口要求绝对合规时，再用枚举约束或在应用层校验最终输出。测试子集的结论只覆盖 138/193 条。要补齐长样本，单独准备 110K 推理数据，用同一个已训练 checkpoint，保留原本训练集计算规则基线。两张卡通过单进程 `device_map auto` 放置模型，实际显存先实测。

```bash
export PREPARED_DATA_DIR=prepared/compact110k
bash run_server.sh prepare --max-length 110000
CUDA_VISIBLE_DEVICES=0,1 TRAIN_FILE=prepared/compact48k/train.jsonl \
  bash run_server.sh eval test base-full-test
CUDA_VISIBLE_DEVICES=0,1 TRAIN_FILE=prepared/compact48k/train.jsonl \
  bash run_server.sh eval test lora-full-test "$CKPT"
unset PREPARED_DATA_DIR
```

## 第二台机器和下一轮数据

1. **先跑独立对照。** 第一台做 48K 主实验，第二台跑基础模型推理、长样本验证，或在固定样本集合下比较工具目录精简方案。先确认 48K 单机的 tokens/s、最长样本显存和验证得分。
2. **再决定 16 卡联合训练。** 单次实验需要加速且两节点 NCCL 网络经过验证时，同步环境、模型和预处理数据，在两台机器上设置同一个 `MASTER_ADDR` 和 `MASTER_PORT`，分别使用 `NODE_RANK=0/1`。训练脚本会在 16 卡把梯度累积设为 1，维持全局 batch 16。ZeRO-3 会跨节点通信，是否更快以实测为准。
3. **补工具判定难例。** 增加同工具异标签、近似工具互斥、信息不足、历史反馈失败、应继续而非 FINISH 的场景。按完整任务轨迹切分，并留出工具或业务域做泛化评估。优先补这批样本，再比较 1 与 2 个 epoch。
4. **增强工具执行和其他能力。** 当前标注只监督候选工具判定。若要提升下一步工具选择、参数生成和多步恢复，补有真实工具调用、参数校验和反馈的轨迹，并定义独立的工具调用成功率和参数正确率。若一个 adapter 还要承担中文问答、提取或推理，再加对应高质量样本与回归集，在验证集比较纯判定训练和约 80/20 的混合训练。混合比例是实验起点，按各任务的实际回归调整。

双机启动示例，两端使用相同代码、数据及模型版本。将地址换为节点 0 的训练网 IP，每台设置自己的 rank。

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.10 MASTER_PORT=29500 \
  bash run_server.sh train
# 在另一台机器执行同一命令，使用 NODE_RANK=1。
```

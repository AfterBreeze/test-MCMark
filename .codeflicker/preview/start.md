# MCMARK 文本生成水印项目运行指南

## 项目概述
这是一个关于 **MCMARK**（多通道无偏水印）的研究项目，用于大语言模型的文本生成水印检测。项目通过划分词汇表并根据水印密钥提升特定段的token概率来实现水印嵌入。

### ⚠️ 运行顺序说明
本项目包含**实验脚本**和**评估脚本**两个步骤，需要先运行实验生成数据，再运行评估分析结果。

---

## 步骤 1：环境准备

在首次运行前，需要先创建Python环境并安装依赖：

```bash
conda create -n mcmark python=3.11
conda activate mcmark
pip install -r requirements.txt
```

---

## 步骤 2：运行水印生成实验

这是**第一步必须运行**的脚本，用于生成带水印的文本：

```bash
bash ./scripts/run_text_generation_exp.sh
```

**脚本说明**：
- 使用模型：`meta-llama/Llama-3.2-3B-Instruct`（可在脚本中修改）
- 数据集：`mmw_book_report`（可在脚本中修改）
- 水印类型：`mcmark`（可在脚本中选择 `main_exp` 或 `mcmark_ablation`）
- 输出目录：`./results/`

**等待标志**：等待脚本执行完成，会在 `./results/` 目录下生成结果文件

```yaml
step1:
  subProjectPath: .
  command: bash ./scripts/run_text_generation_exp.sh
  cwd: .
  port: null
  previewUrl: null
  description: 运行水印生成实验，生成带水印的文本数据
```

---

## 步骤 3：运行评估

实验完成后，运行评估脚本分析水印检测准确率：

```bash
bash ./scripts/run_evaluations.sh
```

**脚本说明**：
- 评估脚本需要指定 `score_path` 参数（已在脚本中预设）
- 计算基线方法和MCMARK的检测准确率
- 假阳性率阈值：`fpr=0.001`

**注意**：确保 `./results/mmw_book_report/Llama_3.2_3B_Instruct/main_exp/score.txt` 文件已存在（由上一步实验生成）

```yaml
step2:
  subProjectPath: .
  command: bash ./scripts/run_evaluations.sh
  cwd: .
  port: null
  previewUrl: null
  description: 运行评估脚本，计算水印检测准确率
```

---

## （可选）步骤 4：运行鲁棒性实验

如果需要测试水印对抗攻击的鲁棒性，运行：

```bash
# 需要先编辑脚本，填入 OpenAI API Key
vim ./scripts/run_robustness_exp.sh  # 修改 openai_api_key 变量

bash ./scripts/run_robustness_exp.sh
```

**支持的攻击类型**：`random_token_replacement`、`gpt_rephrase`、`back_translation`、`dipper`

```yaml
step3_optional:
  subProjectPath: .
  command: bash ./scripts/run_robustness_exp.sh
  cwd: .
  port: null
  previewUrl: null
  description: 运行鲁棒性实验，测试水印对抗攻击的能力
```

---

## 快速开始总结

**标准运行流程**：

```bash
# 1. 激活环境（如未激活）
conda activate mcmark

# 2. 运行实验（生成水印文本）
bash ./scripts/run_text_generation_exp.sh

# 3. 运行评估（分析结果）
bash ./scripts/run_evaluations.sh
```

```yaml
full_pipeline:
  steps:
    - name: 水印生成实验
      command: bash ./scripts/run_text_generation_exp.sh
      depends_on: null
    - name: 评估分析
      command: bash ./scripts/run_evaluations.sh
      depends_on: 水印生成实验
  note: 这是无Web服务的Python研究项目，运行结果为本地文件输出
```

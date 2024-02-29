# ChatOCT
**本科毕设题目：**
基于大语言模型的心血管OCT智能诊断系统

**背景：**

近年来随着我国老龄化社会的加速，罹患急性冠脉综合症的患者人数在不断增加。光学相干断层扫描（**OCT**）对急性冠脉综合征的临床诊断有重要参考价值。然而，我国的OCT专业技术人员严重不足，具备OCT阅片和临床使用经验的主治医生十分有限，远远无法满足日益增长的临床需求。因此，亟需利用人工智能技术推动OCT阅片和诊断的自动化。当前，**大语言模型**在诸多领域展现出巨大潜力。本题目旨在探索将大语言模型与现有的心血管OCT检测**分割模型**结合，实现**面向普通患者**的OCT智能诊断系统。

## 简介

ChatOCT项目旨在将微调的LLM与现有的IVOCT检测模型结合，实现面向普通患者的IVOCT智能诊断系统。项目针对患者提供的IVOCT能够检测出重要病灶，借助微调LLM的医学分析能力和连续对话能力，为患者提供基本的阅片诊疗报告和有效建议。

总体的技术路线是结合G-Swin-Transformer目标检测模型和微调医学LLM为IVOCT影像做医学诊断。具体来说，将IVOCT转码到张量序列作为检测模型的输入，检测模型的输出的视觉特征经过设计的病灶后处理算法转译为文本形式的摘要，客观描述检测到病灶的严重程度，然后嵌入设计的Prompt作为医学LLM的输入，LLM负责整合生成完整的检测报告，以及与用户持续多轮对话。

## 安装

1. 依赖项

     ```
     pip install -r requirements.txt
     ```

2. 安装mmcv（1.2.4~1.4.0）
     使用源码自行构建：https://github.com/open-mmlab/mmcv/tree/v1.3.17

3. 安装mmdet（只能是该版本）

     ```
     pip install -v -e .
     ```

4. ChatGPT连通性测试

     ```shell
     python -m revChatGPT --V3 --api_key $API_KEY
     ```

     API限额查询：https://gptbill.lonlie.cn/

## 数据&权重

**IVOCT输入**：demo/demo.zip

**gswin权重**：checkpoints/gswin_transformer.pth

## 开始

1. 单独OCT检测

     ```
     python OCT_Det/inference_gswin.py
     ```

2. 运行整个项目

     ```
     python demo.py
     ```

     

提供端到端的任务流水线封装，将 “分词器 / 特征提取器 + 模型 + 结果后处理” 整合为一个简单接口，无需手动处理数据预处理和结果解析，可快速实现各类任务的推理。

你只需关注 输入 -> 输出

```python3
from transformers import pipeline

classifier = pipeline('sentiment-analysis")
classifier("I love Transformers!")
```

## 整体架构设计

### Pipeline 核心流程

```
Input
  ↓
Preprocess (Tokenizer / Processor)
  ↓
Model Forward
  ↓
Postprocess
  ↓
Output
```

对应源码中的关键方法：

```python3
Pipeline.__call__()
 ├─ preprocess()
 ├─ _forward()
 ├─ postprocess()
```

### 核心类关系

```
Pipeline (base class)
│
├─ TextClassificationPipeline
├─ TokenClassificationPipeline
├─ QuestionAnsweringPipeline
├─ TextGenerationPipeline
├─ SummarizationPipeline
├─ ImageClassificationPipeline
├─ AutomaticSpeechRecognitionPipeline
└─ ...
```

## ```pipeline()``` 工厂函数详解

### 基本用法

```python
pipeline(
    task,
    model=None,
    tokenizer=None,
    feature_extractor=None,
    image_processor=None,
    device=None,
    framework="pt",
)
```

### 常用参数说明

+ task

  任务名称（字符串）

+ model

  模型名或模型实例

+ tokenizer

  tokenizer 名或实例

+ device

  -1: CPU，0/1/...: GPU

+ framework

  "pt" 或 "tf"

### 自动推断机制

```python3
pipeline("sentiment-analysis")
```

内部会自动：

1. 找到任务默认模型

2. 下载权重

3. 选择 tokenizer

4. 构建对应 Pipeline 子类

## 支持的主要任务类型

### NLP 类

+ text-classification

  TextClassificationPipeline

+ sentiment-analysis

  TextClassificationPipeline

+ token-classification

  TokenClassificationPipeline

+ ner

  TokenClassificationPipeline

+ question-answering

  QuestionAnsweringPipeline

+ summarization

  SummarizationPipeline

+ translation_xx_to_yy

  TranslationPipeline

+ text-generation

  TextGenerationPipeline

+ fill-mask

  FillMaskPipeline

### CV 类
  
| task                   | Pipeline                    |
| ---------------------- | --------------------------- |
| `image-classification` | ImageClassificationPipeline |
| `object-detection`     | ObjectDetectionPipeline     |
| `image-segmentation`   | ImageSegmentationPipeline   |
| `depth-estimation`     | DepthEstimationPipeline     |


### 语音类

| task                           | Pipeline                    |
| ------------------------------ | --------------------------- |
| `automatic-speech-recognition` | ASRPipeline                 |
| `audio-classification`         | AudioClassificationPipeline |


### 多模态

| task                        | Pipeline                       |
| --------------------------- | ------------------------------ |
| `zero-shot-classification`  | ZeroShotClassificationPipeline |
| `visual-question-answering` | VQAPipeline                    |
| `image-to-text`             | ImageToTextPipeline            |

## Pipeline 内部关键方法

### ```preprocess()```

负责把原始输入转换为模型输入格式

### ```_forward()```

调用模型前向传播

### postprocess()

把 logits → 人类可读结果


## 自定义 Pipeline

### 继承 ```Pipeline``

```python3
from transformers import Pipeline

class MyPipeline(Pipeline):
    def preprocess(self, inputs):
        ...

    def _forward(self, model_inputs):
        ...

    def postprocess(self, outputs):
        ...
```

### 注册自定义任务

```python3
PIPELINE_REGISTRY.register_pipeline(
    "my-task",
    pipeline_class=MyPipeline,
    pt_model=MyModel
)
```

  

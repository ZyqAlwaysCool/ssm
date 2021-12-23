# 基于语义与结构双重置信度的三元组质量评估模型

## 环境配置
* Keras
* Python3.6
* [Bert4keras](https://github.com/bojone/bert4keras)
* requirements.txt
## 数据集来源
|数据集名称|链接|
|-|-|
|diakg|[DiaKG: 糖尿病知识图谱数据集](http://openkg.cn/dataset/diakg)|
|diseasekg|[DiseaseKG:基于cnSchma常见疾病信息知识图谱](http://openkg.cn/dataset/disease-information)|
|dbp15k_zh|[JAPE](https://github.com/nju-websoft/JAPE)|
|FB15k|[FB15k](https://paperswithcode.com/dataset/fb15k)|
## 使用
**推荐在jupyter里操作**
|名称|描述|
|-|-|
|`transe_with_classifiers.py`|运行传统机器学习模型|
|`ttmf_process_model.py`|代码参考[KGTtm](https://github.com/TJUNLP/TTMF)模型预处理部分|
|`验证模型效果_xxx.py`|复现模型及结构置信度模块|
|`bert4keras_classifier.py`|语义置信度评估模块，代码参考[Bert4keras](https://github.com/bojone/bert4keras)|
|`dataset`|四个数据集所用到的pkl及原数据|
|`embedding_result`|使用openKE得到的原始embedding|
|`kg_embedding`|处理后使用的embedding|
* dataset及embedding文件提取链接: https://pan.baidu.com/s/1-AFQVbIyIUHlZEe5Q65r0w  密码: ao2p

## 参考
* [KGTtm](https://github.com/TJUNLP/TTMF)
* [OpenKE-PyTorch](https://github.com/thunlp/OpenKE)
* [Bert4Keras](https://github.com/bojone/bert4keras)

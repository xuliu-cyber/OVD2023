# 开放世界目标检测竞赛2023技术方案

## 技术方案介绍

### 算法流程

本算法主要基于two-stage的流程来进行物体的定位与分类。

首先第一阶段为基于Faster RCNN网络对前景目标进行定位。为了提高对新类别物体的检测能力，模型只输出前景物体的定位框，而忽略物体的类别，从而实现class agnostic的检测；

第二阶段为CLIP模型对第一阶段输出的目标进行分类，为了提高分类准确性，使用中文版本的CLIP模型：Chinese CLIP对其进行分类，同时针对图片的来源为电商图片，使用了MUGE数据集对Chinese CLIP模型进行微调。

### 训练框架

本算法基于mmdetection框架以及Chinese CLIP项目。

### 算力规模

本项目使用的环境为Linux，使用一张NVIDIA A40显卡进行训练与测试，CUDA版本为12.0。

## 数据集

本方案所使用的全部数据集包括：初赛和复赛的训练集，以及MUGE数据集。其中初赛和复赛的训练集用于微调Faster RCNN；MUGE数据集用于微调Chinese CLIP模型。

MUGE（Multimodal Understanding and Generation Evaluation）是业界首个大规模中文多模态评测基准，由达摩院联合浙江大学、阿里云天池平台联合发布，中国计算机学会计算机视觉专委会（CCF-CV专委）协助推出。旨在推动多模态表示学习进展，尤其关注多模态预训练。

## 训练及测试

### 环境安装
本算法主要基于mmdetection与Chinese CLIP，首先按照mmdetection与Chinese CLIP的官方教程配置环境。


### 数据集组织
将初赛和复赛的训练集，以及MUGE数据集按照如下方式组织，其中MUGE数据集可从[下载链接](https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/MUGE.zip)进行下载

```
data
├── pretrained_weights/ 
├── experiments/  # 存放微调后的Chinese CLIP模型
├── deploy/	      # 用于存放ONNX & TensorRT部署模型 （本项目用不到）
├── datasets/
│   ├── MUGE/
│   └── .../          # 更多自定义数据集...
│    
├── final/              # 复赛数据集
│   ├── data_final_contest/
│   └── json_final_contest/
└── pre/	            # 初赛数据集
    ├── data_pre_contest/
    └── json_pre_contest/
```

在训练之前，需要先重新生成class agnostic标注文件，即将233个类别标签统一归为1类前景，该标注文件已经提前生成，为./data/pre/json_pre_contest/train_1class.json和./data/final/json_final_contest/train_1class.json文件，可直接使用。

注意：使用自己的数据集微调mmdetection模型时，需要修改mmdetection/mmdet/datasets/coco.py中的classes。

### Faster RCNN微调

本算法依次使用初赛训练集和复赛训练集对Faster RCNN进行微调

首先下载预训练模型[model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210524_124528-26c63de6.pth)
放到./data/pretrained_weights目录下

进入 mmdetection 目录
```
cd mmdetection
```

使用初赛训练集微调Faster RCNN，具体配置文件为[faster-rcnn-config-1.py](mmdetection/faster-rcnn-config-1.py)
```
python tools/train.py faster-rcnn-config-1.py --work-dir workdir/pre
```

使用复赛训练集继续微调，具体配置文件为[faster-rcnn-config-2.py](mmdetection/faster-rcnn-config-2.py)
```
python tools/train.py faster-rcnn-config-2.py --work-dir workdir/final
```

使用微调好的模型对测试集图片进行检测，保存json结果到./final_results目录下，方便后续处理。
```
python demo/image_demo.py ../data/final/data_final_contest/test/ faster-rcnn-config-2.py --weights workdir/final/epoch_1.pth --no-save-vis True
```

### Chinese CLIP模型微调
下载CN-CLIP[ViT-H/14](https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-h-14.pt)预训练模型，放在./data/pretrained_weights目录下。

基于MUGE数据集微调Chinese CLIP模型
```
cd Chinese-CLIP
bash run_scripts/muge_finetune_vit-h-14_rbt-huge.sh
```

### 测试并生成结果文件

```
cd mmdetection
python demo/eval.py
```


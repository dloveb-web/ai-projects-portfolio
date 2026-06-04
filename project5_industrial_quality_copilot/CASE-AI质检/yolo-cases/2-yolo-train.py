#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ultralytics import YOLO

# 从 YAML 配置文件初始化 YOLO11n 模型
model = YOLO("yolov12.yaml")

# 如果有预训练模型，可以使用下面的方式加载
# model = YOLO("model.pt")

# 显示模型信息
model.info()


# In[2]:


# 模型训练
results = model.train(
  data='coco.yaml',
  epochs=1,           # 训练轮数
  batch=256,         # 批次大小
  imgsz=640,         # 输入图像尺寸
  scale=0.5,         # 图像缩放比例 (S:0.9; M:0.9; L:0.9; X:0.9)
  mosaic=1.0,        # 马赛克数据增强概率
  mixup=0.0,         # 混合数据增强概率 (S:0.05; M:0.15; L:0.15; X:0.2)
  copy_paste=0.1,    # 复制粘贴增强概率 (S:0.15; M:0.4; L:0.5; X:0.6)
  device="0",        # 使用的 GPU 设备号
)

# 使用模型进行目标检测
results = model("/root/autodl-tmp/datasets/coco/images/val2017/000000000139.jpg")
results[0].show()  # 显示检测结果


# In[3]:


# 使用model进行目标检测
results = model("/root/autodl-tmp/datasets/coco/images/val2017/000000000139.jpg")
results[0].show()


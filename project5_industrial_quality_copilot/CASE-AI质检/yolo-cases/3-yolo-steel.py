#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ultralytics import YOLO
import os

# 初始化 YOLO 模型
model = YOLO('yolov12.yaml')  # 从配置文件新建模型
# model = YOLO('yolov12n.pt')  # 加载预训练模型（可选）

# 显示模型信息
model.info()

# 模型训练
results = model.train(
    data='./steel_data/dataset.yaml',      # 数据集配置文件
    epochs=100,               # 训练轮数
    batch=16,                 # 批次大小
    imgsz=640,               # 输入图像尺寸
    patience=50,              # 早停轮数
    save=True,               # 保存模型
    device='0',              # GPU设备号，如果使用CPU则设为'cpu'
    workers=8,               # 数据加载线程数
    pretrained=True,         # 是否使用预训练权重
    optimizer='auto',        # 优化器类型
    verbose=True,            # 是否显示详细信息

    # 数据增强参数
    scale=0.5,              # 图像缩放比例
    mosaic=1.0,             # 马赛克数据增强概率
    mixup=0.1,              # 混合数据增强概率
    copy_paste=0.1,         # 复制粘贴增强概率
)

# 在测试集上评估模型
results = model.val()


# In[4]:


# 加载训练好的模型
model2 = YOLO('./runs/detect/train28/weights/best.pt') 
#model2


# In[8]:


# 在单张图像上测试模型
test_image_dir = './steel_data/test/images'
if os.path.exists(test_image_dir):
    for image_name in os.listdir(test_image_dir)[:5]:  # 测试前5张图片
        image_path = os.path.join(test_image_dir, image_name)
        results = model2(image_path)
        # 保存预测结果
        results[0].save(os.path.join('./runs/detect/predict', image_name))


# In[6]:


os.path.join('runs/detect/predict', image_name)


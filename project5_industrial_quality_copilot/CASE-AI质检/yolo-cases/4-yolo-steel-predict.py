#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ultralytics import YOLO
import os

# 加载训练好的模型
model2 = YOLO('./runs/detect/train31/weights/best.pt') 
#model2

# 在单张图像上测试模型
test_image_dir = './steel_data/test/images'
if os.path.exists(test_image_dir):
    for image_name in os.listdir(test_image_dir)[:5]:  # 测试前5张图片
        image_path = os.path.join(test_image_dir, image_name)
        results = model2(image_path)
        # 保存预测结果
        results[0].save(os.path.join('./runs/detect/predict', image_name))


# In[2]:


from ultralytics import YOLO
import os
import pandas as pd
import numpy as np
from PIL import Image

def convert_to_submission_format(results, image_name, img_shape):
    """将YOLO预测结果转换为submission格式"""
    # 获取图像ID（去掉.jpg后缀）
    image_id = int(image_name.split('.')[0])
    
    # 获取预测框、类别和置信度
    boxes = results[0].boxes
    submission_rows = []
    
    if len(boxes) > 0:
        # 获取xyxy格式的边界框（绝对坐标）
        xyxy = boxes.xyxy.cpu().numpy()
        # 获取置信度
        conf = boxes.conf.cpu().numpy()
        # 获取类别
        cls = boxes.cls.cpu().numpy()
        
        # 转换每个预测框
        for box, category_id, confidence in zip(xyxy, cls, conf):
            # 转换为整数坐标
            box = [int(x) for x in box]
            # 创建submission行
            submission_rows.append({
                'image_id': image_id,
                'bbox': str(box),  # 转换为字符串格式
                'category_id': int(category_id),
                'confidence': float(confidence)
            })
    
    return submission_rows

def main():
    # 加载训练好的模型
    model = YOLO('./runs/detect/train31/weights/best.pt')
    
    # 存储所有预测结果
    all_predictions = []
    
    # 在测试集上预测
    test_image_dir = './steel_data/test/images'
    if os.path.exists(test_image_dir):
        # 遍历所有测试图像
        for image_name in os.listdir(test_image_dir):
            if not image_name.endswith('.jpg'):
                continue
                
            # 读取图像以获取尺寸
            img_path = os.path.join(test_image_dir, image_name)
            img = Image.open(img_path)
            img_shape = img.size  # (width, height)
            
            # 进行预测
            results = model(img_path)
            
            # 保存可视化结果
            results[0].save(os.path.join('./runs/detect/predict', image_name))
            
            # 转换为submission格式并添加到列表
            submission_rows = convert_to_submission_format(results, image_name, img_shape)
            all_predictions.extend(submission_rows)
    
    # 创建DataFrame并保存为CSV
    if all_predictions:
        df = pd.DataFrame(all_predictions)
        df.to_csv('submission.csv', index=False)
        print(f"已保存预测结果到 submission.csv，共 {len(df)} 个预测框")
    else:
        print("没有找到任何预测结果")

if __name__ == '__main__':
    main()


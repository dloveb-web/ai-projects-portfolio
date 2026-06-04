#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## 使用yolo12进行预测


# In[1]:


from ultralytics import YOLO

model = YOLO('./yolov12n.pt')
# 使用model进行目标检测
results = model("./000000000139.jpg")
results[0].show()


# In[2]:


# 在验证集上评估
metrics = model.val(data='./coco.yaml', save_json=True)
print(metrics.box.map)  # 打印mAP指标


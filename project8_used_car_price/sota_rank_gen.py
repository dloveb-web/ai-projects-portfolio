# -*- coding: utf-8 -*-
"""
Rank 集成 + 最佳融合
生成: sota_rank_submit.csv（纯Rank）, sota_rank_blend_submit.csv（Rank+融合）
"""

import pandas as pd, numpy as np

BASE_PATH = '/Users/wildon/Desktop/【完成参考】Case-二手车价格预测/'

# 加载 DNN 和融合预测 → 反解出 LightGBM 预测
dnn = pd.read_csv(BASE_PATH + 'sota_dnn_pred.csv')
fusion = pd.read_csv(BASE_PATH + 'sota_dnn_fusion_submit.csv')
sale_ids = fusion['SaleID'].values

dnn_p = dnn['price'].values
fusion_p = fusion['price'].values
lgb_p = 2 * fusion_p - dnn_p  # LGB = 2*fusion - DNN

# Rank 集成
rank_dnn = pd.Series(dnn_p).rank().values
rank_lgb = pd.Series(lgb_p).rank().values
rank_avg = (rank_dnn + rank_lgb) / 2
sorted_fusion = np.sort(fusion_p)
rank_final = np.interp(rank_avg, np.arange(50000), sorted_fusion)
rank_final = np.maximum(rank_final, 50)

# 纯 Rank
pd.DataFrame({'SaleID': sale_ids, 'price': rank_final}).to_csv(
    BASE_PATH + 'sota_rank_submit.csv', index=False)

# Rank + 最佳融合 50/50
blend = 0.5 * rank_final + 0.5 * fusion_p
blend = np.maximum(blend, 50)
pd.DataFrame({'SaleID': sale_ids, 'price': blend}).to_csv(
    BASE_PATH + 'sota_rank_blend_submit.csv', index=False)

print(f"纯Rank: [{rank_final.min():.0f},{rank_final.max():.0f}] mean={rank_final.mean():.0f}")
print(f"Rank融合: [{blend.min():.0f},{blend.max():.0f}] mean={blend.mean():.0f}")
print("✅ sota_rank_submit.csv + sota_rank_blend_submit.csv")

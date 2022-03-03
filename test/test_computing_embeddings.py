# -*- coding:utf-8 -*-
# 对比 numpy 计算相似度 和  sentence_transformers 计算相似度 区别

import numpy as np
import requests
from sentence_transformers.util import cos_sim, semantic_search

url = "http://192.168.1.34:33126/computing_embeddings"
response = requests.request("POST", url, json={"sentences": ["东方红智远三年申购上限"]})
v1 = response.json().get("result")[0]


url = "http://192.168.1.34:33126/computing_embeddings"
response = requests.request("POST", url, json={"sentences": ["东方红稳添裕1号的基金代码"]})
v2 = response.json().get("result")[0]

# sentence_transformers 计算
result = cos_sim(v1, v2)
print(result)

# numpy计算
def hanle_data(result):
    norm_feat = result / np.linalg.norm(result)
    result = norm_feat.tolist()

    return result

result = np.inner(hanle_data(v1), hanle_data(v2))
print(result)
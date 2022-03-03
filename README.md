# text2vec 服务

基于 [text2vec](https://github.com/shibing624/text2vec) 封装函数提供API服务

## API

### 计算句子与文档集之间的相似度值
- Method: **POST**
- Url: /semantic_search
- Body:
```json
{
  "sentences1": ["sentence1", "sentence2", "sentence3"],
  "sentences2": ["sentence1", "sentence2", "sentence3"]
}
```
- Response: 
```json
{
  "result": [
    [
      {"corpus_id": 0, "score": 0.9476608037948608},
      {"corpus_id": 1, "score": 0.9476608037948608},
      {"corpus_id": 2, "score": 0.9476608037948608}
    ],
    [
      {"corpus_id": 0, "score": 0.9476608037948608},
      {"corpus_id": 1, "score": 0.9476608037948608},
      {"corpus_id": 2, "score": 0.9476608037948608}
    ],
    [
      {"corpus_id": 0, "score": 0.9476608037948608},
      {"corpus_id": 1, "score": 0.9476608037948608},
      {"corpus_id": 2, "score": 0.9476608037948608}
    ]
  ]
}
```

### 计算句子之间的相似度值
- Method: **POST**
- Url: /cos_sim
- Body:
```json
{
  "sentences1": ["sentence1", "sentence2", "sentence3"],
  "sentences2": ["sentence1", "sentence2", "sentence3"]
}
```
- Response: 
```json
{
  "result": [
    [0, 1, 1], 
    [0, 1, 1], 
    [0, 1, 1]
  ]
}
```

### 计算句子文本向量
- Method: **POST**
- Url: /computing_embeddings
- Body:
```json
{
  "sentences": ["sentence1", "sentence2", "sentence3"]
}
```
- Response: 
```json
{
  "result": [
    [0, 1, 1], 
    [0, 1, 1], 
    [0, 1, 1]
  ]
}
```

## 环境要求

1. python 3.8+

## 环境变量

| 分组     | 配置项       | 说明                 |
| ------- | ------------| -------------------- |
| 模型参数 | MODEL        | 需要调用的模型名称  默认 paraphrase-multilingual-mpnet-base-v2 小模型 paraphrase-multilingual-MiniLM-L12-v2   |
| 模型参数 | MODELS_PATH  | 所有模型的存储位置     |
| 模型参数 | MODEL_PATH   | 当前服务调用的模型位置  |
| 模型参数 | MODELS_TRAIN | 是否支持个性话模型，默认 False |

## 文件目录说明

```filetree 
├── README.md -- 项目说明
├── run.py -- 程序运行文件
├── run.sh -- 容器运行脚本
├── /requirements.txt -- 项目使用到的依赖包
├── /config.py -- 项目配置文件
├── /init_model.py -- 构建镜像中预加载模型脚本
├── /Dockerfile -- 项目镜像构建文件
```
## 编译
```shell
# 初始化打包
nohup docker build . -f ./Dockerfile -t server.aiknown.cn:31003/ai_service/text2vec_service:master &

# 持续更新
docker build . -f ./Dockerfile_continue -t server.aiknown.cn:31003/ai_service/text2vec_service:master

# push
docker push server.aiknown.cn:31003/ai_service/text2vec_service:master 
```

## 部署/运行

```shell
python run.py
```

or

```shell
docker-compose -f ./compose/ai_service.yml -f ./consumer/dataknown/ai_service.yml -p dataknown --env-file ./env/dataknown_test.env up -d text2vec_service
```

## 使用到的框架

```shell
torch
text2vec
```
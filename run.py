# -*- coding: utf-8 -*-

"""
Documents: https://github.com/shibing624/text2vec
"""
import os

import requests
from flask import request, jsonify, Flask
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim, semantic_search
from config import Config

app = Flask(__name__, root_path=os.getcwd())

personal_model_path = "./models/personalized"  # 个性化模型路径


# 有个性化模型，并且支持模型训练
if os.path.exists(personal_model_path) and Config.MODELS_TRAIN:
    model = SentenceTransformer(personal_model_path)
else:
    model = SentenceTransformer(Config.MODEL_PATH)


@app.route("/token/sync", methods=["POST"])
def post_token_load():
    """
    同步定义的分词信息
    :return:
    """
    token_load()

    return jsonify({"result": True})


@app.route("/token/upload", methods=["POST"])
def post_token_upload():
    """
    同步定义的分词信息
    :return:
    """
    extend_token = request.json.get("token", [])
    token_load(extend_token, False)

    return jsonify({"result": True})


@app.route("/tokenize", methods=["POST"])
def tokenize():
    """
    计算句子与文档集之间的相似度值
    :return:
    """
    global model

    param = {**(request.form or {}), **(request.json or {})}
    sentences = param.pop("sentences")
    return jsonify({"result": model.tokenizer.tokenize(sentences)})


@app.route("/computing_embeddings", methods=["POST"])
def computing_embeddings():
    """
    计算句向量
    :return:
    """
    param = {**(request.form or {}), **(request.json or {})}
    sentences = param.pop("sentences")
    embeddings = model.encode(sentences)

    return jsonify({"result": embeddings.tolist()})


@app.route("/semantic_search", methods=["POST"])
def paraphrase_semantic_search():
    """
    计算句子与文档集之间的相似度值
    :return:
    """
    global model

    param = {**(request.form or {}), **(request.json or {})}
    sentence1 = param.pop("sentence1", None) or param.pop("sentences1", None)
    sentence2 = param.pop("sentence2", None) or param.pop("sentences2", None)
    embeddings1 = model.encode(sentence1)
    embeddings2 = model.encode(sentence2)
    hits = semantic_search(embeddings1, embeddings2, **param)

    return jsonify({"result": hits})


@app.route("/cos_sim", methods=["POST"])
def paraphrase_cos_sim():
    """
    计算句子之间的cos_sim相似度值
    :return:
    """
    param = {**(request.form or {}), **(request.json or {})}
    sentence1 = param.pop("sentence1", None) or param.pop("sentences1", None)
    sentence2 = param.pop("sentence2", None) or param.pop("sentences2", None)

    embeddings1 = model.encode(sentence1)
    embeddings2 = model.encode(sentence2)

    cosine_scores = cos_sim(embeddings1, embeddings2)

    return jsonify({"result": cosine_scores.tolist()})


@app.route("/cluster", methods=["POST"])
def text_cluster():
    """
    计算句子之间的相似度值
    :return:
    """
    param = {**(request.form or {}), **(request.json or {})}

    sentences = param.pop("sentences")  # 句子
    if not sentences or len(sentences) < 3:
        return "sentences 必须传入数据,并且数量大于 3 个!", 500

    min_community_size = param.pop("min_community_size", 2)  # 分组最小数
    threshold = param.pop("threshold", 0.8)  # 阈值
    encode_batch_size = param.pop("encode_batch_size", 32)  # 阈值

    corpus_embeddings = model.encode(
        [""]+sentences, convert_to_tensor=True, batch_size=encode_batch_size
    )

    # fast_clustering
    from sentence_transformers import util

    # 计算分组
    clusters = util.community_detection(
        corpus_embeddings, min_community_size=min_community_size, threshold=threshold
    )
    
    result = []
    for item in clusters:
        if 0 not in item:
            result.append([i-1 for i  in item])

    # 返回索引
    return jsonify({"result": result})


def token_load(extend_token=[], load_from_token_url=True):
    """
    加载个性化词汇重新训练模型

    Args:
        extend_token (list, optional): 外部自定义词汇. Defaults to [].
        load_from_token_url (bool, optional): 是否从远程地址进行获取词汇数据. Defaults to True.
    """
    global model

    tokens = []
    model = SentenceTransformer(Config.MODEL_PATH)

    # 加载分词
    if Config.TOKEN_PATH and os.path.exists(Config.TOKEN_PATH):
        with open(Config.TOKEN_PATH, "r", encoding="utf-8") as file:
            data = file.read().split("\n")
            tokens.extend(data or [])

    # 从远程词库获取数据
    if load_from_token_url and Config.TOKEN_URL:
        response = requests.get(Config.TOKEN_URL)

        if response.ok:
            content = response.json()
            tokens.extend(content.get("data") or [])
        else:
            print("error load token from url")

    # 继承扩展词汇
    tokens.extend(extend_token)

    # 根据大小词汇进行排序
    tokens = list(set(tokens))

    tokens = sorted(set(tokens), key=lambda item: len(str(item or "")), reverse=True)

    # 计算MD5
    md5_file_path = "./models/token_md5.txt"
    last_md5 = None
    if os.path.exists(md5_file_path):
        with open(md5_file_path, "r", encoding="utf-8") as md5_file:
            last_md5 = md5_file.read()

    import hashlib

    md5hash = hashlib.md5()
    md5hash.update((Config.MODEL + "|".join(tokens)).encode("utf-8"))
    md5 = md5hash.hexdigest()
    if md5 == last_md5:
        return

    # 训练
    model.tokenizer.add_tokens(tokens, special_tokens=False)
    model._first_module().auto_model.resize_token_embeddings(len(model.tokenizer))
    import shutil

    # 存在个性化模型，就删除
    if os.path.exists(personal_model_path):
        shutil.rmtree(personal_model_path)

    model.save(personal_model_path)

    model = SentenceTransformer(personal_model_path)
    with open(md5_file_path, "w", encoding="utf-8") as md5_file:
        md5_file.write(md5)
    print("success load token")


# 运行
if __name__ == "__main__":
    print(app.url_map)
    app.run("0.0.0.0", port=5000)

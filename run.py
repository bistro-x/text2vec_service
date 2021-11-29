# -*- coding: utf-8 -*-

"""
Documents: https://github.com/shibing624/text2vec
"""
import os

from flask import request, jsonify, Flask
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim, semantic_search
import requests
from config import Config
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__, root_path=os.getcwd())
model = SentenceTransformer(Config.MODEL_PATH)
model_path = "./models/personalized"  # 个性化模型路径


@app.route("/token/sync", methods=["POST"])
def post_token_load():
    """
    同步定义的分词信息
    :return:
    """
    token_load()

    return jsonify({"result": True})


def token_load():
    global model
    import json

    tokens = []
    model = SentenceTransformer(Config.MODEL_PATH)

    # 加载分词
    if Config.TOKEN_PATH and os.path.exists(Config.TOKEN_PATH):
        with open(Config.TOKEN_PATH, "r") as file:
            data = file.read().split("\n")
            tokens.extend(data or [])

    if Config.TOKEN_URL:
        save_file_path = "token_url.json"  # 保存文件
        response = requests.get(Config.TOKEN_URL)

        if response.ok:
            content = response.json()
            tokens.extend(content.get("data") or [])
            # 写入缓存文件
            with open(save_file_path, "w") as token_file:
                json.dump(content.get("data"), token_file)
        else:
            print("error load token from url")

            # 读取缓存文件
            if os.path.exists(save_file_path):
                with open(save_file_path, "r") as token_file:
                    data = json.load(token_file)
                    print(data)
                    tokens.extend(data)
                    print("load temp file")

    tokens = sorted(set(tokens), key=lambda item: len(item), reverse=True)
    # print(tokens)
    model.tokenizer.add_tokens(tokens, special_tokens=False)
    model._first_module().auto_model.resize_token_embeddings(len(model.tokenizer))
    model.tokenizer.save_pretrained(model_path)
    print("success load token")


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


@app.route("/semantic_search", methods=["POST"])
def paraphrase_semantic_search():
    """
    计算句子与文档集之间的相似度值
    :return:
    """
    global model

    param = {**(request.form or {}), **(request.json or {})}
    sentences1 = param.pop("sentences1")
    sentences2 = param.pop("sentences2")
    embeddings1 = model.encode(sentences1)
    embeddings2 = model.encode(sentences2)
    hits = semantic_search(embeddings1, embeddings2, **param)

    return jsonify({"result": hits})


@app.route("/cos_sim", methods=["POST"])
def paraphrase_cos_sim():
    """
    计算句子之间的相似度值
    :return:
    """
    param = {**(request.form or {}), **(request.json or {})}
    sentences1 = param.pop("sentences1")
    sentences2 = param.pop("sentences2")
    embeddings1 = model.encode(sentences1)
    embeddings2 = model.encode(sentences2)
    cosine_scores = cos_sim(embeddings1, embeddings2)

    return jsonify({"result": cosine_scores.tolist()})


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


# 加载已有分词
if os.path.exists(model_path):
    model.tokenizer = AutoTokenizer.from_pretrained(model_path)

# 运行
if __name__ == "__main__":
    app.run("0.0.0.0", port=5000)

# -*- coding: utf-8 -*-

"""
Documents: https://github.com/shibing624/text2vec
"""
import os
import time

from flask import request, jsonify, Flask
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.util import cos_sim, semantic_search
import requests
from config import Config
from transformers import AutoTokenizer, AutoModel
import shutil
import schedule
from concurrent.futures.thread import ThreadPoolExecutor

app = Flask(__name__, root_path=os.getcwd())

personal_model_path = "./models/personalized"  # 个性化模型路径

# 加载已有分词
if os.path.exists(personal_model_path):
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
        response = requests.get(Config.TOKEN_URL)

        if response.ok:
            content = response.json()
            tokens.extend(content.get("data") or [])
        else:
            print("error load token from url")

    tokens = sorted(set(tokens), key=lambda item: len(item), reverse=True)
    model.tokenizer.add_tokens(tokens, special_tokens=True)
    model._first_module().auto_model.resize_token_embeddings(len(model.tokenizer))
    model.save(personal_model_path)
    model = SentenceTransformer(personal_model_path)

    print("success load token")


def test():
    print(time.time())


def auto_token_load():
    schedule.every().minutes.do(test)
    if Config.AUTO_TOKEN:
        schedule.every().days.at("00:00").do(token_load)
    while True:
        schedule.run_pending()


# 运行
if __name__ == "__main__":
    ThreadPoolExecutor(2).submit(work)
    app.run("0.0.0.0", port=5000)

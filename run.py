"""
Documents: https://github.com/shibing624/text2vec
"""
import os

from flask import request, jsonify, Flask
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim, semantic_search
import requests
from config import Config

app = Flask(__name__, root_path=os.getcwd())
model = SentenceTransformer(Config.MODEL_PATH)


@app.route("/token/sync", methods=["POST"])
def post_token_load():
    """
    同步定义的分词信息
    :return:
    """

    token_load()
    return jsonify({"result": True})


def token_load():
    # 加载分词
    if Config.TOKEN_PATH and os.path.exists(Config.TOKEN_PATH):
        with open(Config.TOKEN_PATH, "r") as file:
            data = file.readlines()
            model.tokenizer.add_tokens(data, special_tokens=True)

    if Config.TOKEN_URL:
        response = requests.get(Config.TOKEN_URL)
        if response.ok:
            content = response.json()
            model.tokenizer.add_tokens(content.get("data"), special_tokens=True)
        else:
            print(f"请求分词服务发生错误:{response.message}")

    print("成功加载数据")


@app.route("/semantic_search", methods=["POST"])
def paraphrase_semantic_search():
    """
    计算句子与文档集之间的相似度值
    :return:
    """
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


if __name__ == "__main__":
    token_load()
    app.run("0.0.0.0", port=5000)

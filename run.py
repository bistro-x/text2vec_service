"""
Documents: https://github.com/shibing624/text2vec
"""
import os

from flask import request, jsonify, Flask
from sentence_transformers.util import cos_sim, semantic_search, SentenceTransformer

from config import Config

app = Flask(__name__, root_path=os.getcwd())
model = SentenceTransformer(Config.MODEL_PATH)

# 加载分词
if Config.get("TOKEN_PATH") and os.path.exists(Config.get("TOKEN_PATH")):
    with open(Config.get("TOKEN_PATH"), "r") as file:
        data = file.readlines()
        model.tokenizer.add_tokens(data, special_tokens=False)

if Config.get("TOKEN_URL"):
    response = request.get(Config.get("TOKEN_URL"))
    if response.ok:
        content = response.json()
        model.tokenizer.add_tokens(content.get("data"), special_tokens=False)
    else:
        print(f"请求分词服务发生错误:{response.message}")


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
    app.run("0.0.0.0", port=5000)

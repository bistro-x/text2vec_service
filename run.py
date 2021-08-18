"""
Documents: https://github.com/shibing624/text2vec
"""
import os

from flask import request, jsonify, Flask
from text2vec import SBert
from sentence_transformers.util import cos_sim, semantic_search

app = Flask(__name__, root_path=os.getcwd())
model = SBert("./models/paraphrase-multilingual-MiniLM-L12-v2")


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


if __name__ == "__main__":
    app.run("0.0.0.0", port=5000)

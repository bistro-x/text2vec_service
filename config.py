import os


class Config:
    """
    配置信息
    """

    AUTO_TOKEN = False
    MODEL = os.getenv("MODEL", "paraphrase-multilingual-mpnet-base-v2")  # 调用的模型
    MODELS_PATH = os.getenv("MODELS_PATH", "./models")  # 模型位置
    MODELS_TRAIN = bool(os.getenv("MODELS_TRAIN", False))  # 支持模型训练

    MODEL_PATH = os.path.join(MODELS_PATH, MODEL)

    TOKEN_PATH = os.getenv("TOKEN_PATH", "./models/token.txt")  # 词汇文件
    TOKEN_URL = os.getenv(
        "TOKEN_URL"
    )  # 词汇文件的接口返回

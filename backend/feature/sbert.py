# 文件名：sbert.py
# 作用：使用 Sentence-BERT 提取文本的语义向量（支持 GPU）

import torch
from sentence_transformers import SentenceTransformer


class SBERTFeature:
    """
    SBERT 语义向量提取类
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        初始化 SBERT 模型
        :param model_name: 预训练 SBERT 模型名称（支持中文）
        """
        # 自动选择 GPU / CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载预训练模型
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: list):
        """
        将文本列表编码为语义向量
        :param texts: 文本列表（未分词的原始文本也可以）
        :return: 向量矩阵
        """
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        return embeddings

    def encode_single(self, text: str):
        """
        编码单条文本
        """
        return self.encode([text])[0]

# 文件名：tfidf.py
# 作用：使用 TF-IDF 方法将文本转换为向量表示
#TfidfVectorizer：专门用于将文本转换为TF-IDF特征矩阵的类
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfFeature:
    """
    TF-IDF 特征提取类
    """

    def __init__(self, max_features: int = 5000):
        """
        初始化 TF-IDF 向量器
        :param max_features: 最大特征词数量
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            token_pattern=r"(?u)\b\w+\b"
        )

    def fit_transform(self, texts: list):
        """
        训练并转换文本为 TF-IDF 特征
        :param texts: 文本列表
        :return: TF-IDF 特征矩阵
        """
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: list):
        """
        使用已训练好的模型转换新文本
        :param texts: 文本列表
        :return: TF-IDF 特征矩阵
        """
        return self.vectorizer.transform(texts)

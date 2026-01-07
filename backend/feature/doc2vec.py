# 文件名：doc2vec.py
# 作用：使用 Doc2Vec 方法将文本转换为文档向量

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class Doc2VecFeature:
    """
    Doc2Vec 文档向量特征提取类
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 40
    ):
        """
        初始化 Doc2Vec 模型参数
        :param vector_size: 向量维度
        :param window: 上下文窗口大小
        :param min_count: 最小词频
        :param epochs: 训练轮数
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None

    def train(self, texts: list):
        """
        训练 Doc2Vec 模型
        :param texts: 分词后的文本列表（每个元素是字符串，词之间用空格）
        """
        # 构建 TaggedDocument
        documents = [
            TaggedDocument(words=text.split(), tags=[i])
            for i, text in enumerate(texts)
        ]

        # 初始化 Doc2Vec 模型
        self.model = Doc2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            epochs=self.epochs
        )

        # 构建词表
        self.model.build_vocab(documents)

        # 训练模型
        self.model.train(
            documents,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs
        )

    def infer_vector(self, text: str):
        """
        推断单条新文本的向量
        :param text: 分词后的文本
        :return: 文档向量
        """
        if self.model is None:
            raise ValueError("Doc2Vec 模型尚未训练")

        return self.model.infer_vector(text.split())

    def get_document_vectors(self):
        """
        获取训练语料中所有文档的向量
        :return: 文档向量列表
        """
        if self.model is None:
            raise ValueError("Doc2Vec 模型尚未训练")

        return [self.model.dv[i] for i in range(len(self.model.dv))]

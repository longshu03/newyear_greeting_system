# 文件名：test_doc2vec.py
# 作用：测试 Doc2Vec 文档向量模块是否正常工作

from feature.doc2vec import Doc2VecFeature

if __name__ == "__main__":
    # 模拟已经清洗和分词后的文本
    texts = [
        "张三 重要 项目 技术 攻关 系统 开发",
        "李四 团队 管理 项目 推进 绩效 提升",
        "王五 技术 创新 系统 架构 设计"
    ]

    # 初始化 Doc2Vec 特征提取器
    doc2vec = Doc2VecFeature(vector_size=50, epochs=30)

    # 训练模型
    doc2vec.train(texts)

    # 获取训练文本的向量
    vectors = doc2vec.get_document_vectors()

    print("文档向量数量：", len(vectors))
    print("单个文档向量维度：", len(vectors[0]))

    # 推断新文本向量
    new_text = "技术 项目 系统 优化"
    new_vector = doc2vec.infer_vector(new_text)

    print("新文本向量维度：", len(new_vector))

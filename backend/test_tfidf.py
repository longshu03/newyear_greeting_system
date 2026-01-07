#用于测试tfidf特征提取模块
from feature.tfidf import TfidfFeature
if __name__=="__main__":
    texts=[
        "张三 重要 项目 技术 攻关",
        "李四 团队 管理 项目 推进",
        "王五 技术 创新 系统 开发"
    ]

    #创建TF-IDF特征提取器
    tfidf=TfidfFeature(max_features=20)

    #训练并转换
    tfidf_matrix=tfidf.fit_transform(texts)

    print("TF-IDF特征矩阵形状为:",tfidf_matrix.shape)
    print("TF-IDF特征矩阵为：")
    #将稀疏矩阵转换为普通数组并打印
    print(tfidf_matrix.toarray())
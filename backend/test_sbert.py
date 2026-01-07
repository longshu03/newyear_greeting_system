# 文件名：test_sbert.py
# 作用：测试 SBERT 语义向量模块是否正常工作（GPU）

from feature.sbert import SBERTFeature
import torch

if __name__ == "__main__":
    # 初始化 SBERT
    sbert = SBERTFeature()

    # 测试文本
    texts = [
        "张三在2024年完成了多个重要技术项目。",
        "李四在团队管理和项目推进方面表现突出。",
        "王五在系统架构设计和技术创新方面贡献显著。"
    ]

    # 编码文本
    embeddings = sbert.encode(texts)

    print("是否使用 GPU：", embeddings.is_cuda)
    print("向量数量：", embeddings.shape[0])
    print("向量维度：", embeddings.shape[1])

    # 单条文本测试
    single_vec = sbert.encode_single("技术创新推动系统优化")
    print("单条向量维度：", single_vec.shape[0])

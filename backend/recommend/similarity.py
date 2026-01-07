# 文件名：similarity.py
# 作用：向量相似度计算（兼容 TF-IDF / Doc2Vec / SBERT）

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import issparse


def cosine_similarity_torch(query_vec, candidate_vecs):
    """
    使用 PyTorch 计算余弦相似度（适用于稠密向量）
    """
    if not isinstance(query_vec, torch.Tensor):
        query_vec = torch.tensor(query_vec, dtype=torch.float32)

    if not isinstance(candidate_vecs, torch.Tensor):
        candidate_vecs = torch.tensor(candidate_vecs, dtype=torch.float32)

    query_vec = query_vec.unsqueeze(0)  # [1, dim]
    query_norm = torch.nn.functional.normalize(query_vec, dim=1)
    cand_norm = torch.nn.functional.normalize(candidate_vecs, dim=1)

    scores = torch.mm(query_norm, cand_norm.T).squeeze(0)
    return scores.detach().cpu().numpy()


def cosine_similarity_sklearn(query_vec, candidate_vecs):
    """
    使用 sklearn 计算余弦相似度（适用于 TF-IDF 稀疏矩阵）
    """
    query_vec = query_vec.reshape(1, -1)
    scores = cosine_similarity(query_vec, candidate_vecs)[0]
    return scores


def top_k_similarity(query_vec, candidate_vecs, k):
    """
    自动判断向量类型，选择合适的相似度计算方式
    """
    # ⭐ 关键判断：TF-IDF 稀疏矩阵
    if issparse(candidate_vecs):
        scores = cosine_similarity_sklearn(query_vec, candidate_vecs)
    else:
        scores = cosine_similarity_torch(query_vec, candidate_vecs)

    return scores

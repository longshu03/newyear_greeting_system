# 文件名：test_similarity.py
# 作用：测试相似度计算与推荐模块

import torch
from feature.sbert import SBERTFeature
from recommend.similarity import top_k_similarity

if __name__ == "__main__":
    # 初始化 SBERT
    sbert = SBERTFeature()

    # 模拟贺词模板
    templates = [
        "在过去一年中，您在技术创新方面表现突出，为公司发展作出了重要贡献。",
        "您在团队管理和项目推进方面成绩显著，是团队的重要支柱。",
        "您在系统架构设计和技术攻关中展现了卓越的专业能力。"
    ]

    # 模拟员工报告
    employee_report = "在2024年，我主要负责系统架构设计和核心技术攻关工作。"

    # 编码模板和员工报告
    template_vecs = sbert.encode(templates)
    report_vec = sbert.encode_single(employee_report)

    # 计算 Top-2 相似模板
    top_indices = top_k_similarity(report_vec, template_vecs, k=2)

    print("推荐模板索引：", top_indices)
    print("推荐结果：")
    for idx in top_indices:
        print("-", templates[idx])

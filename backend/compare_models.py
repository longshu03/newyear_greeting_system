"""
文件名：compare_models.py
作用：对比 TF-IDF / Doc2Vec / SBERT 三种模型的模板选择与生成效果

使用方法：
1. 将员工年度总结放在 data/reports/
2. 修改 FILE_NAME
3. 运行 python compare_models.py
"""

import os
from pipeline.greeting_pipeline import GreetingPipeline

# =========================
# 配置区域
# =========================
FILE_NAME = "6.docx"                     # 待处理报告文件
OUTPUT_DIR = "data/generated_greetings"  # 输出目录
FEATURE_TYPES = ["tfidf", "doc2vec", "sbert"]

# =========================
# 主流程
# =========================
def main():
    print("\n========== 模型对比实验开始 ==========\n")

    for feature in FEATURE_TYPES:
        print(f"\n【当前模型】：{feature.upper()}")
        print("-" * 40)

        pipeline = GreetingPipeline(feature_type=feature)
        result = pipeline.run_file(FILE_NAME, output_dir=OUTPUT_DIR)

        if result:
            print(f"匹配模板文件: {result['template_name']}")
            print(f"相似度得分: {result['similarity_score']}")
            print("\n生成的新春贺词：\n")
            print(result["greeting"])
        else:
            print("❌ 生成失败")

        print("\n" + "=" * 40)

    print("\n========== 模型对比实验结束 ==========\n")


if __name__ == "__main__":
    main()

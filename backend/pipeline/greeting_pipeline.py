# 文件名：greeting_pipeline.py
# 作用：基于模板 + NLP 特征匹配 +（可选）LLM 的个性化新春贺词生成系统

import os
import numpy as np

from preprocess.parser import parse_pdf, parse_docx
from preprocess.cleaner import load_stopwords, clean_text

from feature.tfidf import TfidfFeature
from feature.doc2vec import Doc2VecFeature
from feature.sbert import SBERTFeature

from recommend.similarity import top_k_similarity
from generate.filler import fill_template

# LLM 模块
from LLM.qwen_model import QWenGenerator

# 员工姓名解析
from profile.employee_profile import EmployeeProfileManager

# ⭐ 个性化核心模块
from analysis.report_analyzer import ReportAnalyzer
from profile.employee_profile import EmployeeProfileBuilder


class GreetingPipeline:
    """
    新春贺词生成系统
    模式一：模板 + NLP 特征匹配（不使用 LLM）
    模式二：模板 + 员工画像 + LLM 个性化生成
    """

    def __init__(self, feature_type="sbert", use_llm=False):
        self.feature_type = feature_type.lower()
        self.use_llm = use_llm

        # 员工姓名解析器
        self.employee_profile_manager = EmployeeProfileManager()

        # ⭐ 新增：总结分析 & 员工画像构建
        self.report_analyzer = ReportAnalyzer()
        self.profile_builder = EmployeeProfileBuilder()

        # 加载停用词
        self.stopwords = load_stopwords("preprocess/stopwords.txt")

        # 加载模板
        self.templates = self._load_templates()

        # 初始化特征模型并向量化模板
        self.template_vectors = self._init_feature_model()

        # 初始化 LLM（可选）
        if self.use_llm:
            self.llm = QWenGenerator()

    # ---------------------------------------------------
    # 模板与特征处理
    # ---------------------------------------------------

    def _load_templates(self):
        """
        加载贺词模板（name + content）
        """
        template_dir = "data/templates"
        templates = []

        for file in os.listdir(template_dir):
            if file.endswith(".txt"):
                with open(os.path.join(template_dir, file), "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        templates.append({
                            "name": file,
                            "content": content
                        })
        return templates

    def _init_feature_model(self):
        """
        初始化特征模型，并对模板进行向量化
        """
        texts = [t["content"] for t in self.templates]

        if self.feature_type == "tfidf":
            self.feature_model = TfidfFeature()
            return self.feature_model.fit_transform(texts)

        elif self.feature_type == "doc2vec":
            self.feature_model = Doc2VecFeature()
            return self.feature_model.train(texts)

        elif self.feature_type == "sbert":
            self.feature_model = SBERTFeature()
            return self.feature_model.encode(texts)

        else:
            raise ValueError("不支持的特征类型")

    def _encode_report(self, text):
        """
        对员工年度总结进行向量化
        """
        if self.feature_type == "tfidf":
            return self.feature_model.transform([text])[0]

        elif self.feature_type == "doc2vec":
            return self.feature_model.infer_vector(text)

        elif self.feature_type == "sbert":
            return self.feature_model.encode([text])[0]

    # ---------------------------------------------------
    # 核心流程
    # ---------------------------------------------------

    def run_file(self, file_name, output_dir="data/generated_greetings"):
        """
        对单个员工年度总结生成新春贺词
        """
        file_path = os.path.join("data/reports", file_name)

        # 1️⃣ 文本解析
        if file_name.endswith(".pdf"):
            raw_text = parse_pdf(file_path)
        elif file_name.endswith(".docx"):
            raw_text = parse_docx(file_path)
        else:
            raise ValueError("仅支持 PDF / DOCX 文件")

        # 2️⃣ 文本清洗
        cleaned_text = clean_text(raw_text, self.stopwords)

        # 3️⃣ 员工姓名解析
        employee_name, name_source = self.employee_profile_manager.get_employee_name(file_name)

        # ---------------------------------------------------
        # ⭐ 个性化关键步骤：总结分析 → 员工画像
        # ---------------------------------------------------
        analysis_result = self.report_analyzer.extract(cleaned_text)
        keywords = self.report_analyzer.extract_keywords(cleaned_text)
        achievements = self.report_analyzer.extract_key_sentences(cleaned_text)

        employee_profile = self.profile_builder.build(
            name=employee_name,
            keywords=keywords,
            achievements=achievements
        )

        # ---------------------------------------------------
        # 模板匹配（两种模式共用）
        # ---------------------------------------------------

        report_vector = self._encode_report(cleaned_text)
        scores = top_k_similarity(report_vector, self.template_vectors, k=len(self.templates))
        best_idx = int(np.argmax(scores))
        best_template = self.templates[best_idx]

        base_info = {
            "姓名": employee_name,
            "年份": "2024",
            "关键词": "、".join(keywords) if keywords else "本职工作"
        }

        base_greeting = fill_template(best_template["content"], base_info)

        # ---------------------------------------------------
        # 模式一：不使用 LLM
        # ---------------------------------------------------

        if not self.use_llm:
            greeting = base_greeting

        # ---------------------------------------------------
        # 模式二：LLM 深度个性化
        # ---------------------------------------------------

        else:
            prompt = f"""
你是医院信息化部的一名员工，请以同事的角度，为员工撰写中国新春贺词。

【员工姓名】
{employee_profile['姓名']}

【年度关键词】
{", ".join(employee_profile['年度关键词'])}

【年度主要工作成果】
"""  # 成果列表
            for i, s in enumerate(employee_profile["年度成果"], 1):
                prompt += f"{i}. {s}\n"

            prompt += f"""
【基础贺词参考】
{base_greeting}

【写作要求】（必须严格遵守）：
1. 贺词第一行必须是：“{employee_name}同志：”
2. 内容必须基于上述年度成果，不得虚构或添加其他建议
3. 语言正式、庄重、积极，符合机关单位风格
4. 字数控制在 80–120 字
5. 不得出现工号、表格字段、年份
6. 只输出贺词正文，不要解释、不加标题
7. 体现总结成绩 + 新年激励
8、完全不要提供任何考核建议或额外信息

请开始生成贺词：
"""

            greeting = self.llm.generate_greeting(prompt).strip()

        # ---------------------------------------------------
        # 后处理校验
        # ---------------------------------------------------
        if not greeting.startswith(f"{employee_name}同志："):
            greeting = f"{employee_name}同志：\n" + greeting
        if greeting.endswith(employee_name):
            greeting = greeting[: -len(employee_name)].strip()
        """if employee_name not in greeting:
            greeting = f"{employee_name}同志：\n\n" + greeting
        """

        # ---------------------------------------------------
        # 保存结果
        # ---------------------------------------------------

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{employee_name}_greeting.txt")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(greeting)

        print(f"✅ 贺词已生成并保存：{output_file}")
        print(f"👤 员工姓名：{employee_name}")
        print(f"📍 姓名来源：{'映射表' if name_source == 'map' else '文件名回退'}")
        print(f"🧠 使用 LLM：{self.use_llm}")

        return {
            "greeting": greeting,
            "employee_name": employee_name,
            "employee_name_source": name_source,
            "template": best_template["name"],
            "used_keywords": keywords,
            "used_achievements": achievements,
            "feature_type": self.feature_type,
            "use_llm": self.use_llm,
            "output_file": output_file
        }

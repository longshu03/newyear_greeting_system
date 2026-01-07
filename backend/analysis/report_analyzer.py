# analysis/report_analyzer.py

import re
from sklearn.feature_extraction.text import TfidfVectorizer


class ReportAnalyzer:
    """
    从年度总结中提取“可用于个性化贺词”的关键信息
    """

    # ---------------------------
    # 公有方法：供外部直接调用
    # ---------------------------

    def extract_keywords(self, text, top_k=6):
        """
        返回文本的关键词列表（公有方法）
        """
        return self._extract_keywords(text, top_k=top_k)

    def extract_key_sentences(self, text):
        """
        返回文本中的关键句子列表（公有方法）
        """
        return self._extract_key_sentences(text)

    def extract(self, text):
        """
        返回结构化分析结果，包含关键词、关键句、岗位/角色推断
        """
        return {
            "keywords": self._extract_keywords(text),
            "key_sentences": self._extract_key_sentences(text),
            "role_summary": self._infer_role(text)
        }

    # ---------------------------
    # 私有方法：核心算法
    # ---------------------------

    def _extract_keywords(self, text, top_k=6):
        """
        使用 TF-IDF 提取关键词（私有方法）
        """
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2)
        )
        tfidf = vectorizer.fit_transform([text])
        scores = tfidf.toarray()[0]
        words = vectorizer.get_feature_names_out()

        idx = scores.argsort()[-top_k:][::-1]
        return [words[i] for i in idx]

    def _extract_key_sentences(self, text):
        """
        提取关键句子（私有方法）
        """
        sentences = re.split(r"[。！？\n]", text)
        verbs = ["负责", "主导", "完成", "推进", "建设", "优化", "保障", "实现"]

        result = []
        for s in sentences:
            if any(v in s for v in verbs) and len(s) > 12:
                result.append(s.strip())

        return result[:5]

    def _infer_role(self, text):
        """
        根据关键字简单推断员工岗位/角色
        """
        if "系统" in text or "平台" in text:
            return "信息系统建设与运维"
        if "保障" in text or "维护" in text:
            return "信息保障支持"
        if "开发" in text:
            return "系统开发建设"
        return "综合信息化支持"

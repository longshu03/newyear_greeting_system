
#导入python类型提示模块
from typing import Dict


class GreetingGenerator:

    def build_prompt(self, profile, year):
        prompt = f"""
你是一名企业行政文案专家，请为员工生成正式的新春贺词。

【员工信息】
姓名：{profile['姓名']}
年份：{year}

【年度关键词】
{", ".join(profile['年度关键词'])}

【年度主要成果】
"""
        for i, s in enumerate(profile["年度成果"], 1):
            prompt += f"{i}. {s}\n"

        prompt += """
要求：
1. 内容必须基于上述信息
2. 不得虚构事实
3. 语言正式、积极
"""
        return prompt

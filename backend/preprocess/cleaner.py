# 作用：对解析后的文本进行脱敏、清洗和中文分词

import jieba   #仲中文分词库，将连续的中文句子切分成词语
import re      #正则表达式库，用于文本匹配和清洗
import random

#1、停用词加载
def load_stopwords(stopword_path: str) -> set:
    """
    加载停用词表
    """
    stopwords = set()
    with open(stopword_path, "r", encoding="utf-8") as f:
        for line in f:
            stopwords.add(line.strip())
    return stopwords

"""2、隐私脱敏模块
    def anonymize_text(text:str)->str:

    对文本中的敏感信息进行脱敏处理
    
    #身份证号脱敏
    #匹配18位身份证号
    def replace_id(match):
        return f"<ID_{random.randint(10000000,99999999)}>"
    text=re.sub(r'\b\d{17}[\dXx]\b',replace_id,text)

    #手机号码脱敏
    #匹配11位手机号
    def replace_phone(match):
        return f"PHONE_{random.randint(100000, 999999)}"

    text = re.sub(
        r'(手机号|电话|联系电话)[:：]?\s*1[3-9]\d{9}',
        replace_phone,
        text
    )


    #员工姓名脱敏
    name_patterns = [
        r'(姓名|员工|申报人)[:：]\s*([\u4e00-\u9fa5]{2,3})',
        r'([\u4e00-\u9fa5]{2,3})\s*同志'
    ]
    for p in name_patterns:
        text = re.sub(p, r'\1：员工', text)

    #公司名称脱敏
    text = re.sub(
        r'[\u4e00-\u9fa5]+(有限公司|集团|研究院|医院|中心|妇幼保健院)',
        '某单位',
        text
    )

    return text
    #地点信息脱敏
    location_pattern = (
        r'(北京|上海|天津|重庆|河北省|山西省|辽宁省|吉林省|黑龙江省|'
        r'江苏省|浙江省|安徽省|福建省|江西省|山东省|河南省|湖北省|湖南省|'
        r'广东省|海南省|四川省|贵州省|云南省|陕西省|甘肃省|青海省|'
        r'内蒙古自治区|广西壮族自治区|西藏自治区|宁夏回族自治区|新疆维吾尔自治区)'
    )
    text = re.sub(location_pattern, '某地', text)
"""

#3、文本清洗+分词
def clean_text(text: str, stopwords: set) -> str:
    """
    文本清洗与分词
    """
    # 先做隐私脱敏
    #text = anonymize_text(text)

    # 去除非中文、字母、数字的符号
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)

    # 中文分词（讲中文句子切分成词语列表）
    words = jieba.lcut(text)

    # 去停用词和过短词
    words = [w for w in words if w not in stopwords and len(w) > 1]

    return " ".join(words)

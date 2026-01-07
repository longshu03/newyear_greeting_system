# 文件名：filler.py
# 作用：将推荐的贺词模板与员工信息进行安全填充

#导入正则表达式
import re


def fill_template(template: str, info: dict) -> str:
    """
    模板填充函数

    :param template: 模板文本（包含 {姓名} 等占位符）
    :param info: 员工信息字典
    :return: 填充后的完整贺词
    """

    filled_text = template

    # ===== 1. 统一定义可用字段 =====
    allowed_fields = {
        "姓名": info.get("姓名", "该员工"),
        "年份": info.get("年份", ""),
        "领域": info.get("关键词", "相关工作")
    }

    # ===== 2. 逐个字段替换 =====
    for key, value in allowed_fields.items():
        filled_text = filled_text.replace(f"{{{key}}}", str(value))

    # ===== 3. 清理可能残留的占位符 =====
    filled_text = re.sub(r"\{.*?\}", "", filled_text)

    return filled_text

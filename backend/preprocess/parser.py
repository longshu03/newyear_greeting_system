# 作用：解析不同格式的员工年度报告，统一提取文本内容

from docx import Document
import pdfplumber


def parse_txt(file_path: str) -> str:
    """
    解析 txt 文件
    :param file_path: txt 文件路径
    :return: 文本内容
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def parse_docx(file_path: str) -> str:
    """
    解析 docx 文件
    :param file_path: docx 文件路径
    :return: 文本内容
    """
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def parse_pdf(file_path: str) -> str:
    """
    解析 pdf 文件
    :param file_path: pdf 文件路径
    :return: 文本内容
    """
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def parse_file(file_path: str) -> str:
    """
    根据文件后缀自动选择解析方式
    """
    if file_path.endswith(".txt"):
        return parse_txt(file_path)
    elif file_path.endswith(".docx"):
        return parse_docx(file_path)
    elif file_path.endswith(".pdf"):
        return parse_pdf(file_path)
    else:
        raise ValueError("不支持的文件格式")

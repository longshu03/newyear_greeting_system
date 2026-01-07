# 文件名：main.py
# 作用：FastAPI 后端服务入口，整合新春贺词智能生成完整流程

from fastapi import FastAPI
from pydantic import BaseModel

# 导入系统模块
from feature.sbert import SBERTFeature
from recommend.similarity import top_k_similarity
from generate.greeting_generator import GreetingGenerator
from data.templates import TEMPLATES

# 创建 FastAPI 应用
app = FastAPI(
    title="新春贺词智能生成系统",
    description="基于语义相似度的新春贺词智能生成后端服务",
    version="1.0"
)

# 初始化核心组件（只加载一次，避免重复耗时）
sbert = SBERTFeature()
generator = GreetingGenerator()

# 对模板进行一次性编码
template_vectors = sbert.encode(TEMPLATES)


# ======================
# 请求数据模型
# ======================
class GreetingRequest(BaseModel):
    姓名: str
    年份: str
    员工报告: str


# ======================
# 接口定义
# ======================
@app.post("/generate_greeting")
def generate_greeting(request: GreetingRequest):
    """
    新春贺词生成接口
    """
    # 1. 对员工报告进行语义编码
    report_vector = sbert.encode_single(request.员工报告)

    # 2. 计算与模板的相似度，选 Top-1
    top_index = top_k_similarity(
        report_vector,
        template_vectors,
        k=1
    )[0]

    # 3. 获取最相似的模板
    best_template = TEMPLATES[top_index]

    # 4. 构造员工信息字典
    employee_info = {
        "姓名": request.姓名,
        "年份": request.年份,
        "关键词": "相关工作"
    }

    # 5. 生成新春贺词
    greeting = generator.generate(best_template, employee_info)

    # 6. 返回结果
    return {
        "推荐模板索引": top_index,
        "生成的新春贺词": greeting
    }


# ======================
# 测试接口
# ======================
@app.get("/")
def root():
    return {"message": "新春贺词智能生成系统后端运行正常"}


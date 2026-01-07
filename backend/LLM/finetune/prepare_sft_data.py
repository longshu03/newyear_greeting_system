# prepare_sft_data.py
# 作用：把员工总结 + 最终贺词，整理成指令微调数据

import json
import os
import random

def build_sample(employee_name, summary, greeting):
    return {
        "instruction": "根据员工年度总结生成正式的新春贺词",
        "input": f"员工姓名：{employee_name}\n年度总结：{summary}",
        "output": greeting
    }

def main():
    samples = []

    # ===== 示例（你之后会用真实数据替换）=====
    samples.append(
        build_sample(
            employee_name="张三",
            summary="2024年主要负责医院信息系统运维与升级，保障系统稳定运行。",
            greeting="张三同志：过去一年你在医院信息系统建设与运维工作中认真负责，为保障医院信息化稳定运行作出了积极贡献。新的一年，愿你再接再厉，取得更大成绩。"
        )
    )

    output_path = "backend/LLM/finetune/sft_dataset.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"✅ 指令微调数据已生成：{output_path}")

if __name__ == "__main__":
    main()

# 文件名：test_greeting_generator.py
# 作用：测试贺词模板生成模块

from generate.greeting_generator import GreetingGenerator

if __name__ == "__main__":
    generator = GreetingGenerator()

    # 贺词模板
    template = (
        "{姓名}在{年份}年中，在{关键词}方面表现突出，"
        "为公司发展作出了重要贡献，特此表示感谢与祝贺！"
    )

    # 员工信息
    employee_info = {
        "姓名": "张三",
        "年份": "2024",
        "关键词": "系统架构设计和技术攻关"
    }

    greeting = generator.generate(template, employee_info)

    print("生成的新春贺词：")
    print(greeting)

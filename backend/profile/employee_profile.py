import json
import os


class EmployeeProfileManager:
    """
    员工画像管理器
    - 文件名 → 员工身份映射
    - 后续可扩展：岗位、部门、关键词、风格偏好等
    """

    def __init__(self):
        self.map_path = os.path.join(
            os.path.dirname(__file__),
            "employee_map.json"
        )
        self.employee_map = self._load_map()

    def _load_map(self):
        if not os.path.exists(self.map_path):
            print("⚠ 未找到 employee_map.json，使用默认规则")
            return {}

        with open(self.map_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_employee_name(self, file_name: str):
        """
        根据文件名获取员工姓名,返回：姓名+是否来自映射表
        """
        if file_name in self.employee_map:
            return self.employee_map[file_name],"map"
        else:
            return os.path.splitext(file_name)[0],"fallback"
            
#新增（个性化）
class EmployeeProfileBuilder:

    def build(self, name, keywords, achievements):
        return {
            "姓名": name,
            "年度关键词": keywords,
            "年度成果": achievements
        }

import os
import sys
import argparse
import json
import requests

# 尝试导入pandas库，用于读取Excel文件
try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False

# 将项目的根目录添加到Python的模块搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试导入geopy库
try:
    from geopy.geocoders import Nominatim
    has_geopy = True
except ImportError:
    has_geopy = False

# 尝试导入不同的LLM模型
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    from langchain_core.output_parsers import StrOutputParser
    from langchain_community.chat_models import ChatTongyi, ChatOpenAI, ChatAnthropic
    has_langchain = True
except ImportError:
    has_langchain = False

# 从 rag_builder 模块获取知识库检索器（使用标准接口）
try:
    from tools import query_knowledge_base, query_knowledge_base_as_text
    has_rag = True
except ImportError:
    has_rag = False

# 备用城市经纬度映射（当geopy不可用时使用）
BACKUP_CITY_COORDINATES = {
    "乌兰察布": (40.98, 113.12),
    "北京": (39.90, 116.41),
    "上海": (31.23, 121.47),
    "广州": (23.13, 113.26),
    "深圳": (22.55, 114.05),
    "杭州": (30.27, 120.15),
    "成都": (30.67, 104.06),
    "武汉": (30.59, 114.31),
    "西安": (34.34, 108.94),
    "南京": (32.06, 118.78)
}

# 城市到省份的映射
CITY_TO_PROVINCE = {
    "乌兰察布": "内蒙古",
    "北京": "北京",
    "上海": "上海",
    "广州": "广东",
    "深圳": "广东",
    "杭州": "浙江",
    "成都": "四川",
    "武汉": "湖北",
    "西安": "陕西",
    "南京": "江苏"
}

# 省份与碳排因子的映射（2023年数据）
PROVINCE_CARBON_FACTOR = {
    "北京": 0.5554,
    "天津": 0.6796,
    "河北": 0.6516,
    "山西": 0.6634,
    "内蒙古": 0.6479,
    "辽宁": 0.4878,
    "吉林": 0.4671,
    "黑龙江": 0.5229,
    "上海": 0.5737,
    "江苏": 0.5827,
    "浙江": 0.4974,
    "安徽": 0.6553,
    "福建": 0.4211,
    "江西": 0.5836,
    "山东": 0.6191,
    "河南": 0.5897,
    "湖北": 0.4044,
    "湖南": 0.4976,
    "广东": 0.4419,
    "广西": 0.4476,
    "海南": 0.3648,
    "重庆": 0.5581,
    "四川": 0.1564,
    "贵州": 0.5683,
    "云南": 0.1333,
    "陕西": 0.6335,
    "甘肃": 0.4471,
    "青海": 0.1796,
    "宁夏": 0.6187,
    "新疆": 0.6021
}

class DataCenterAgent1:
    """
    Agent 1: 数据中心绿电消纳规划设计顾问
    负责接收用户输入，获取地理位置相关数据，生成标准化数据包
    """
    
    def __init__(self):
        """
        初始化 Agent 1（不再维护 RAG，直接使用标准接口）
        """
        pass

    def get_coordinates(self, city_name):
        """
        使用geopy获取城市经纬度
        
        参数:
            city_name: 城市名称
            
        返回:
            tuple: (纬度, 经度)
        """
        if has_geopy:
            try:
                # 初始化地理编码器
                geolocator = Nominatim(user_agent="my_data_center_project")
                # 获取地理位置
                location = geolocator.geocode(city_name)
                
                if location:
                    print(f"🌍 使用geopy获取{city_name}的经纬度")
                    return location.latitude, location.longitude
                else:
                    print(f"⚠️ geopy未找到{city_name}，使用备用数据")
            except Exception as e:
                print(f"❌ geopy调用失败: {e}，使用备用数据")
        else:
            print("⚠️ geopy库不可用，使用备用数据")
        
        # 如果geopy不可用或失败，使用备用城市经纬度映射
        if city_name in BACKUP_CITY_COORDINATES:
            return BACKUP_CITY_COORDINATES[city_name]
        else:
            # 如果在备用映射中也找不到，使用北京的经纬度
            print(f"⚠️ 未找到{city_name}的经纬度，使用默认值（北京）")
            return 39.90, 116.41
    
    def get_province_from_coordinates_opencage(self, lat, lon):
        """
        使用OpenCage API从经纬度获取省份
        
        参数:
            lat: 纬度
            lon: 经度
            
        返回:
            dict: 包含省份、城市等信息的字典，失败返回None
        """
        # 优先使用环境变量中的API密钥
        api_key = os.getenv("OPENCAGE_API_KEY")
        
        # 如果环境变量中没有，使用默认API密钥
        if not api_key:
            api_key = '51d955cb1a324735ab6dee041e09002c'
            print("⚠️ 使用默认OpenCage API密钥")
        
        url = "https://api.opencagedata.com/geocode/v1/json"
        
        params = {
            'q': f"{lat},{lon}",
            'key': api_key,
            'language': 'zh',  # 返回中文结果
            'pretty': 1
        }
        
        print(f"🌐 调用OpenCage API获取省份信息")
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # 检查请求是否成功
            data = response.json()
            
            if data['results']:
                # 直接从components中获取省份
                components = data['results'][0]['components']
                province = components.get('state')  # 省份
                city = components.get('city') or components.get('town') or components.get('village')
                
                print(f"🌍 完整地址: {data['results'][0]['formatted']}")
                print(f"✅ OpenCage API获取省份成功: {province}")
                return {
                    'province': province,
                    'city': city,
                    'country': components.get('country'),
                    'coordinates': {
                        'lat': data['results'][0]['geometry']['lat'],
                        'lng': data['results'][0]['geometry']['lng']
                    }
                }
            else:
                print("⚠️ OpenCage API未找到结果")
                return None
        except Exception as e:
            print(f"❌ OpenCage API调用失败: {e}")
            return None
    
    def get_province(self, city_name):
        """
        获取城市对应的省份
        
        参数:
            city_name: 城市名称
            
        返回:
            str: 省份名称（不带"省"字）
        """
        # 1. 首先尝试使用OpenCage API（通过经纬度）
        print(f"🔍 开始获取{city_name}的省份信息")
        latitude, longitude = self.get_coordinates(city_name)
        opencage_result = self.get_province_from_coordinates_opencage(latitude, longitude)
        
        if opencage_result and opencage_result.get('province'):
            province = opencage_result['province']
            # 去掉省份名称中的"省"字，确保与PROVINCE_CARBON_FACTOR字典匹配
            if province.endswith('省'):
                province = province[:-1]
            print(f"✅ 从OpenCage API获取省份: {province}")
            return province
        else:
            print("⚠️ OpenCage API获取省份失败")
        
        # 2. 如果OpenCage API失败，尝试使用geopy
        if has_geopy:
            try:
                # 初始化地理编码器
                geolocator = Nominatim(user_agent="data_center_project/your_email@example.com")
                # 添加请求频率限制
                from geopy.extra.rate_limiter import RateLimiter
                geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
                # 查询城市（限定中国区域）
                location = geocode(f"{city_name}, China")
                
                if location:
                    # 完整地址解析结果
                    print(f"🌍 完整地址: {location.address}")
                    print(f"🌍 纬度: {location.latitude}, 经度: {location.longitude}")
                    
                    # 从地址中提取省份
                    address_parts = location.address.split(',')
                    # 中国地址格式通常是：市, 省, 中国
                    # 例如："杭州市, 浙江省, 中国" → 省份是第二个元素
                    if len(address_parts) >= 2:
                        province = address_parts[1].strip()
                        # 去掉省份名称中的"省"字，确保与PROVINCE_CARBON_FACTOR字典匹配
                        if province.endswith('省'):
                            province = province[:-1]
                        print(f"✅ 从geopy地址中提取省份: {province}")
                        return province
                else:
                    print(f"⚠️ geopy未找到城市: {city_name}")
            except Exception as e:
                print(f"❌ geopy调用失败: {e}")
        else:
            print("⚠️ geopy库不可用")
        
        # 3. 如果OpenCage API和geopy都失败，使用备用城市到省份的映射
        if city_name in CITY_TO_PROVINCE:
            province = CITY_TO_PROVINCE[city_name]
            print(f"⚠️ 使用备用映射获取省份: {province}")
            return province
        else:
            # 如果在备用映射中也找不到，返回默认值（北京）
            print(f"⚠️ 未找到{city_name}对应的省份，使用默认值（北京）")
            return "北京"
    
    def get_carbon_emission_factor(self, city_name):
        """
        获取指定城市的碳排因子
        
        参数:
            city_name: 城市名称
            
        返回:
            float: 碳排因子（kgCO₂/kWh）
        """
        # 1. 获取城市对应的省份
        province = self.get_province(city_name)
        print(f"📋 {city_name} 属于 {province}")
        
        # 2. 从省份与碳排因子映射中获取碳排因子
        if province in PROVINCE_CARBON_FACTOR:
            carbon_factor = PROVINCE_CARBON_FACTOR[province]
            print(f"✅ 从映射中获取碳排因子: {carbon_factor} kgCO₂/kWh")
            return carbon_factor
        else:
            # 如果找不到省份对应的碳排因子，返回默认值
            print(f"⚠️ 未找到{province}对应的碳排因子，使用默认值 0.5")
            return 0.5
    
    def get_environmental_data(self, location):
        """
        获取指定地理位置的环境数据
        包括：年均温度、年均风速、年均日照时长、碳排放因子
        
        参数:
            location: 地理位置（城市名）
            
        返回:
            dict: 环境数据
        """
        # 1. 获取城市经纬度
        latitude, longitude = self.get_coordinates(location)
        print(f"✅ 找到{location}的经纬度: {latitude}, {longitude}")
        
        # 2. 调用Open-Meteo API获取环境数据
        try:
            # 构建API URL
            api_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date=2025-01-01&end_date=2026-03-09&daily=sunshine_duration,wind_speed_10m_mean,temperature_2m_mean"
            print(f"🌤️  调用Open-Meteo API: {api_url}")
            
            # 发送请求
            response = requests.get(api_url)
            response.raise_for_status()  # 检查请求是否成功
            
            # 解析响应数据
            data = response.json()
            
            # 3. 计算年均值
            # 提取每日数据
            daily_data = data.get('daily', {})
            
            # 计算年均温度（摄氏度）
            temp_values = daily_data.get('temperature_2m_mean', [])
            annual_temperature = sum(temp_values) / len(temp_values) if temp_values else 10.0
            
            # 计算年均风速（米/秒，API返回的是km/h，需要转换）
            wind_values = daily_data.get('wind_speed_10m_mean', [])
            # 转换为米/秒: 1 km/h = 0.277778 m/s
            wind_values_mps = [v * 0.277778 for v in wind_values]
            annual_wind_speed = sum(wind_values_mps) / len(wind_values_mps) if wind_values_mps else 4.0
            
            # 计算年均日照时长（小时，API返回的是秒，需要转换）
            sunshine_values = daily_data.get('sunshine_duration', [])
            # 转换为小时: 1小时 = 3600秒
            sunshine_values_hours = [v / 3600 for v in sunshine_values]
            annual_sunshine_hours = sum(sunshine_values_hours) if sunshine_values_hours else 2500
            
            # 4. 获取碳排因子
            carbon_factor = self.get_carbon_emission_factor(location)
            
            # 5. 构建返回数据
            environmental_data = {
                "annual_temperature": round(annual_temperature, 2),
                "annual_wind_speed": round(annual_wind_speed, 2),
                "annual_sunshine_hours": round(annual_sunshine_hours, 2),
                "carbon_emission_factor": carbon_factor
            }
            
            print(f"✅ 成功获取{location}的环境数据:")
            print(f"  - 年均温度: {environmental_data['annual_temperature']}°C")
            print(f"  - 年均风速: {environmental_data['annual_wind_speed']} m/s")
            print(f"  - 年均日照时长: {environmental_data['annual_sunshine_hours']} 小时")
            
            return environmental_data
            
        except Exception as e:
            print(f"❌ 调用Open-Meteo API失败: {e}")
            # 如果API调用失败，返回默认值，但使用真实的碳排因子
            carbon_factor = self.get_carbon_emission_factor(location)
            return {
                "annual_temperature": 10.0,
                "annual_wind_speed": 4.0,
                "annual_sunshine_hours": 2500,
                "carbon_emission_factor": carbon_factor
            }
    
    def get_electricity_price(self, location):
        """
        获取指定地理位置的电价数据
        从knowledge_base的202603各省电价第一列数据中获取D-I 6个数据
        
        参数:
            location: 地理位置（城市名）
            
        返回:
            dict: 电价数据，包含D-I 6个数据
        """
        # 1. 获取城市对应的省份
        province = self.get_province(location)
        print(f"📋 {location} 属于 {province}")
        
        # 2. 尝试从Excel文件中读取电价数据
        if has_pandas:
            try:
                # 构建Excel文件路径
                excel_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "excel", "202603各省电价xlsx.xlsx")
                print(f"📊 读取电价数据文件: {excel_path}")
                
                # 读取Excel文件
                df = pd.read_excel(excel_path)
                

                
                # 尝试从第二列查找对应省份的行
                province_row = df[df.iloc[:, 1] == province]
                
                if not province_row.empty:
                    # 获取D-I列的数据（假设D列是第4列，I列是第9列，Python索引从0开始）
                    d_col = province_row.iloc[0, 3] if len(province_row.columns) > 3 else None
                    e_col = province_row.iloc[0, 4] if len(province_row.columns) > 4 else None
                    f_col = province_row.iloc[0, 5] if len(province_row.columns) > 5 else None
                    g_col = province_row.iloc[0, 6] if len(province_row.columns) > 6 else None
                    h_col = province_row.iloc[0, 7] if len(province_row.columns) > 7 else None
                    i_col = province_row.iloc[0, 8] if len(province_row.columns) > 8 else None
                    
                    # 构建返回数据
                    electricity_data = {
                        "尖峰电价": d_col,
                        "高峰电价": e_col,
                        "平段电价": f_col,
                        "低谷电价": g_col,
                        "深谷电价": h_col,
                        "最大峰谷价差": i_col
                    }
                    
                    print(f"✅ 从Excel文件获取电价数据成功")
                    print(f"  - 尖峰电价: {d_col}")
                    print(f"  - 高峰电价: {e_col}")
                    print(f"  - 平段电价: {f_col}")
                    print(f"  - 低谷电价: {g_col}")
                    print(f"  - 深谷电价: {h_col}")
                    print(f"  - 最大峰谷价差: {i_col}")
                    
                    return electricity_data
                else:
                    print(f"⚠️ 未找到{province}的电价数据")
            except Exception as e:
                print(f"❌ 读取Excel文件失败: {e}")
        else:
            print("⚠️ pandas库不可用")
        
        # 如果Excel文件读取失败，返回默认值
        print("⚠️ 使用默认电价数据")
        return {
            "尖峰电价": 0.5,  # 元/度
            "高峰电价": 0.4,  # 元/度
            "平段电价": 0.3,  # 元/度
            "低谷电价": 0.25,  # 元/度
            "深谷电价": 0.2,  # 元/度
            "最大峰谷价差": 0.15  # 元/度
        }
    
    def process_user_input(self, user_input):
        """
        处理用户输入，生成标准化数据包（增强 RAG 知识检索）
        
        参数:
            user_input: dict，包含用户输入的所有参数
            
        返回:
            dict: 标准化数据包
        """
        # 提取用户输入参数
        location = user_input.get('location', '')
        business_type = user_input.get('business_type', '通用')
        planned_area = user_input.get('planned_area', 0)  # 平方米
        planned_load = user_input.get('planned_load', 0)  # kW
        Computing_power_density = user_input.get('Computing_power_density', 0)  # kW/机柜
        priority = user_input.get('priority', '环保型')
        green_energy_target = user_input.get('green_energy_target', 90)  # %
        pue_target = user_input.get('pue_target', 1.2)
        budget_constraint = user_input.get('budget_constraint', 0)  # 万元
        
        # 获取环境数据
        environmental_data = self.get_environmental_data(location)
        
        # 获取电价数据
        electricity_price = self.get_electricity_price(location)
        
        # 生成标准化数据包
        standardized_data = {
            "project_info": {
                "location": location,
                "business_type": business_type,
                "planned_area": planned_area,
                "planned_load": planned_load,
                "Computing_power_density": Computing_power_density,
                "priority": priority,
                "green_energy_target": green_energy_target,
                "pue_target": pue_target,
                "budget_constraint": budget_constraint
            },
            "environmental_data": environmental_data,
            "electricity_price": electricity_price,
            "timestamp": "2026-03-10"  # 实际项目中应使用当前时间
        }
        
        return standardized_data
    
    def generate_data_packet(self, user_input):
        """
        生成标准结构的规范化数据包
        
        参数:
            user_input: dict，包含用户输入的所有参数
            
        返回:
            str: JSON格式的数据包
        """
        standardized_data = self.process_user_input(user_input)
        return json.dumps(standardized_data, ensure_ascii=False, indent=2)


# ============================================================
# Agent 1 LangGraph节点函数 (适配统一状态)
# ============================================================

# 导入统一状态类型
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from graph import GreenDataCenterState


def requirement_analysis_node(state: dict) -> dict:
    """
    Agent 1: 需求与约束解析专家 - LangGraph Node
    
    接收用户输入，解析需求，获取环境数据。
    这是LangGraph工作流的第一个节点。
    
    参数:
        state: GreenDataCenterState类型，包含以下键:
            - user_requirements: 用户需求字典
            
    返回:
        更新后的state，新增/更新以下字段:
            - user_requirements: 解析后的用户需求
            - environmental_data: 环境数据（气候、碳排因子等）
            - electricity_price: 电价数据（已转换为标准格式）
            - iteration_count: 迭代计数器（如果是重试则+1）
    """
    print("\n" + "="*60)
    print("📋 [Agent 1: 需求与约束解析专家] 开始工作")
    print("="*60)
    
    # 获取用户需求
    user_requirements = state.get('user_requirements', {})

    # 创建Agent实例并处理
    agent1 = DataCenterAgent1()
    
    # 获取环境数据
    location = user_requirements.get('location', '')
    environmental_data = agent1.get_environmental_data(location)
    
    # 添加经纬度信息到环境数据
    latitude, longitude = agent1.get_coordinates(location)
    province = agent1.get_province(location)
    environmental_data['latitude'] = latitude
    environmental_data['longitude'] = longitude
    environmental_data['province'] = province
    
    print(f"\n🌡️ 环境数据:")
    print(f"  - 位置: {province} ({latitude:.2f}°N, {longitude:.2f}°E)")
    print(f"  - 年均温度: {environmental_data.get('annual_temperature', 'N/A')}°C")
    print(f"  - 年均风速: {environmental_data.get('annual_wind_speed', 'N/A')} m/s")
    print(f"  - 年均日照: {environmental_data.get('annual_sunshine_hours', 'N/A')} 小时")
    print(f"  - 碳排因子: {environmental_data.get('carbon_emission_factor', 'N/A')} kgCO₂/kWh")
    
    # 获取电价数据（保持中文格式，无需转换）
    electricity_price = agent1.get_electricity_price(location)
    print(f"\n💰 电价数据:")
    print(f"  - 尖峰电价：{electricity_price['尖峰电价']} 元/kWh")
    print(f"  - 高峰电价：{electricity_price['高峰电价']} 元/kWh")
    print(f"  - 平段电价：{electricity_price['平段电价']} 元/kWh")
    print(f"  - 低谷电价：{electricity_price['低谷电价']} 元/kWh")
    print(f"  - 深谷电价：{electricity_price['深谷电价']} 元/kWh")
    print(f"  - 最大峰谷价差：{electricity_price['最大峰谷价差']} 元/kWh")
    
    # 更新迭代计数器（如果是重试流程）
    iteration_count = state.get('iteration_count', 0)
    if iteration_count > 0:
        print(f"\n🔄 这是第 {iteration_count} 次参数调整")
    
    print("\n" + "="*60)
    print("✅ [Agent 1: 需求与约束解析专家] 工作完成")
    print("="*60)
    
    # 返回更新后的状态（保持原有字段，更新新字段）
    return {
        **state,  # 保留原有状态
        "user_requirements": user_requirements,
        "environmental_data": environmental_data,
        "electricity_price": electricity_price,
        "iteration_count": iteration_count
    }



# --- 主程序入口 (用于独立测试) ---
if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Agent 1: 数据中心绿电消纳规划设计顾问")
    parser.add_argument("--location", type=str, default="乌兰察布", help="数据中心地理位置")
    parser.add_argument("--business-type", type=str, default="通用", help="业务类型")
    parser.add_argument("--planned-area", type=float, default=10000, help="计划面积（平方米）")
    parser.add_argument("--planned-load", type=float, default=5000, help="计划负荷（kW）")
    parser.add_argument("--Computing_power-density", type=float, default=8, help="算力密度（kW/机柜）")
    parser.add_argument("--priority", type=str, default="环保型", help="优先级（可靠、经济、环保型）")
    parser.add_argument("--green-energy-target", type=float, default=90, help="绿电目标（%）")
    parser.add_argument("--pue-target", type=float, default=1.2, help="PUE目标")
    parser.add_argument("--budget-constraint", type=float, default=10000, help="预算约束（万元）")
    args = parser.parse_args()
    
    print("===== 开始测试 Agent 1: 数据中心绿电消纳规划设计顾问 =====")
    
    # 创建Agent 1实例
    agent1 = DataCenterAgent1()
    
    # 准备用户输入
    user_input = {
        'location': args.location,
        'business_type': args.business_type,
        'planned_area': args.planned_area,
        'planned_load': args.planned_load,
        'Computing_power_density': args.Computing_power_density,
        'priority': args.priority,
        'green_energy_target': args.green_energy_target,
        'pue_target': args.pue_target,
        'budget_constraint': args.budget_constraint
    }
    
    print("\n--- 用户输入参数 ---")
    for key, value in user_input.items():
        print(f"{key}: {value}")
    
    # 生成标准化数据包
    data_packet = agent1.generate_data_packet(user_input)
    
    print("\n--- 生成的标准化数据包 ---")
    print(data_packet)
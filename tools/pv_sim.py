import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Literal
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults # 可选：用于更精准的搜索
from geopy.geocoders import Nominatim
import pvlib
from pvlib.location import Location
from pvlib.pvsystem import PVSystem, FixedMount
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import math

# ==========================================
# 1. 内置经验数据库 (作为 API 失败的兜底)
# ==========================================
EMPIRICAL_DATA = {
    "guiyang": {"lat": 26.65, "lon": 106.72, "altitude": 1100, "timezone": "Asia/Shanghai", "peak_sun_hours": 3.8, "temp_summer": 24},
    "beijing": {"lat": 39.90, "lon": 116.40, "altitude": 50, "timezone": "Asia/Shanghai", "peak_sun_hours": 4.5, "temp_summer": 30},
    "lasa": {"lat": 29.65, "lon": 91.11, "altitude": 3650, "timezone": "Asia/Shanghai", "peak_sun_hours": 6.2, "temp_summer": 22},
    "shanghai": {"lat": 31.23, "lon": 121.47, "altitude": 4, "timezone": "Asia/Shanghai", "peak_sun_hours": 3.9, "temp_summer": 32},
    "default": {"lat": 30.0, "lon": 120.0, "altitude": 50, "timezone": "UTC", "peak_sun_hours": 4.0, "temp_summer": 25}
}

def _get_location_info(city: str) -> Dict[str, Any]:
    """
    获取地点信息：优先通过 geopy 地理编码查询，失败则查内置库。
    """
    city_key = city.lower().replace(" ", "").replace("市", "")

    # 1. 优先 geopy 查询经纬度
    try:
        geolocator = Nominatim(user_agent="green_data_center_pv_sim")
        query_candidates = [city, f"{city}, China", city.replace("市", "")]

        for query in query_candidates:
            location = geolocator.geocode(query, timeout=8)
            if location is not None:
                return {
                    "lat": float(location.latitude),
                    "lon": float(location.longitude),
                    "altitude": 50,
                    "timezone": "Asia/Shanghai",
                    "peak_sun_hours": EMPIRICAL_DATA["default"]["peak_sun_hours"],
                    "temp_summer": EMPIRICAL_DATA["default"]["temp_summer"],
                }
    except Exception as e:
        print(f"[Warning] Geopy geocoding failed: {e}. Falling back to empirical DB.")
    
    # 2. 检查内置库
    if city_key in EMPIRICAL_DATA:
        return EMPIRICAL_DATA[city_key]
    
    # 3. 模糊匹配 (简单实现)
    for key, data in EMPIRICAL_DATA.items():
        if key in city_key or city_key in key:
            return data
            
    # 4. 返回默认值
    print(f"[Warning] City '{city}' not found by geopy or DB. Using default coordinates.")
    default_data = EMPIRICAL_DATA["default"].copy()
    return default_data

def _fetch_weather_data(lat: float, lon: float, date_str: str) -> Optional[pd.DataFrame]:
    """
    调用 Open-Meteo 免费 API 获取历史/典型气象数据 (GHI, Temp)。
    如果 API 失败，返回 None，触发兜底逻辑。
    """
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": "shortwave_radiation,temperature_2m",
            "timezone": "auto"
        }
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if "hourly" not in data:
            return None
            
        df = pd.DataFrame({
            "time": pd.to_datetime(data["hourly"]["time"]),
            "ghi": data["hourly"]["shortwave_radiation"],
            "temp_air": data["hourly"]["temperature_2m"]
        })
        df.set_index("time", inplace=True)
        return df
    except Exception as e:
        print(f"[Warning] Weather API failed: {e}. Switching to synthetic model.")
        return None

def _generate_synthetic_weather(location: Location, date_str: str) -> pd.DataFrame:
    """
    兜底方案：基于天文模型生成典型晴天的合成气象数据。
    使用 pvlib 的 clearsky 模型。
    """
    times = pd.date_range(start=date_str, end=f"{date_str} 23:59", freq="H", tz=location.tz)
    
    # 计算晴空辐照度 (Clear Sky GHI)
    cs = location.get_clearsky(times)
    
    # 合成温度：基于季节的简单估算 (实际应更复杂)
    month = int(date_str.split("-")[1])
    # 简单模拟：夏季高温，冬季低温
    base_temp = 25 if 5 <= month <= 9 else 10
    temp_series = pd.Series([base_temp + 5 * (1 - abs(12 - t.hour)/12) for t in times], index=times)
    
    df = pd.DataFrame({
        "ghi": cs["ghi"],
        "dni": cs["dni"],
        "dhi": cs["dhi"],
        "temp_air": temp_series
    })
    return df

@tool
def generate_pv_profile(
    city: str, 
    date: str = "2024-06-21", 
    capacity_kw: float = 1.0, 
    tilt: float = None, 
    azimuth: float = 180
) -> Dict[str, Any]:
    """
    生成指定城市在特定日期的光伏出力曲线。
    
    该工具结合了实时气象API检索与pvlib物理仿真引擎。
    如果无法获取真实天气数据，将自动切换至'典型晴天'天文模型进行估算。
    
    参数:
    - city: 城市名称 (例如: "贵阳", "Beijing", "拉萨")
    - date: 日期字符串 (YYYY-MM-DD). 默认为夏至日 (典型晴天).
    - capacity_kw: 装机容量 (kWp). 默认为 1.0 (返回系数).
    - tilt: 组件倾角 (度). 默认为当地纬度.
    - azimuth: 组件方位角 (度). 0=正北, 180=正南. 默认为 180.
    
    返回:
    - 包含逐小时出力 (kW), 辐照度 (W/m²), 温度 (°C) 及汇总统计的 JSON 对象.
    """
    
    # 1. 获取地理位置信息
    loc_info = _get_location_info(city)
    location = Location(
        latitude=loc_info["lat"],
        longitude=loc_info["lon"],
        tz=loc_info["timezone"],
        altitude=loc_info["altitude"],
        name=city
    )
    
    # 自动设置最佳倾角 (如果未指定)
    if tilt is None:
        tilt = loc_info["lat"]
        
    # 2. 获取气象数据 (真实 API 或 合成数据)
    weather_df = _fetch_weather_data(loc_info["lat"], loc_info["lon"], date)
    
    if weather_df is None:
        data_source = "Synthetic Clear-Sky Model"
        print(f"Using synthetic clear-sky model for {city} on {date}.")
        weather_df = _generate_synthetic_weather(location, date)
    else:
        data_source = "Open-Meteo API"
        # Open-Meteo 在 timezone=auto 时可能返回 naive 时间，先本地化再转换
        if weather_df.index.tz is None:
            weather_df = weather_df.tz_localize(location.tz)
        else:
            weather_df = weather_df.tz_convert(location.tz)
        
    # 3. 构建光伏系统模型
    system = PVSystem(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        module_parameters={"pdc0": capacity_kw * 1000, "gamma_pdc": -0.004}, # 标准单晶硅
        inverter_parameters={"pdc0": capacity_kw * 1000, "eta_inv_nom": 0.96},
        temperature_model_parameters=TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_polymer"],
        modules_per_string=1,
        strings_per_inverter=1
    )
    
    # 4. 执行仿真计算链 (Modeling Chain)
    try:
        # 计算太阳位置并推导斜面辐照度 (POA)
        solar_position = location.get_solarposition(weather_df.index)
        irradiance_components = pvlib.irradiance.erbs(
            ghi=weather_df["ghi"],
            zenith=solar_position["zenith"],
            datetime_or_doy=weather_df.index,
        )

        poa = system.get_irradiance(
            solar_zenith=solar_position["apparent_zenith"],
            solar_azimuth=solar_position["azimuth"],
            dni=irradiance_components["dni"],
            ghi=weather_df["ghi"],
            dhi=irradiance_components["dhi"],
        )
        
        # 计算组件温度
        cell_temp = system.get_cell_temperature(poa["poa_global"], weather_df["temp_air"], 1.0, model="sapm")
        
        # 计算直流功率
        dc_power = system.pvwatts_dc(poa["poa_global"], cell_temp)
        
        # 计算交流功率 (最终出力)
        ac_power = pvlib.inverter.pvwatts(
            pdc=dc_power,
            pdc0=capacity_kw * 1000,
            eta_inv_nom=0.96,
        )
        
        # pvlib/pvwatts 输出单位为 W，这里统一换算为 kW
        ac_power = (ac_power / 1000.0).clip(lower=0).fillna(0)
        
    except Exception as e:
        # 极端降级：如果 pvlib 计算失败，返回简单的正弦波
        print(f"[Critical] PVLib calculation failed: {e}. Using fallback sine wave.")
        hours = range(24)
        # 简单正弦模拟
        ac_power = pd.Series([max(0, capacity_kw * 0.9 * math.sin(math.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0 for h in hours], 
                             index=weather_df.index[:24])

    # 5. 格式化输出
    result_data = []
    total_energy = 0
    
    for t, power in ac_power.items():
        hour_val = round(power, 3) # kW
        total_energy += hour_val # 近似积分 (步长1h)
        
        result_data.append({
            "hour": t.hour,
            "timestamp": t.isoformat(),
            "power_kw": hour_val,
            "coefficient": round(hour_val / capacity_kw, 3) if capacity_kw > 0 else 0
        })
        
    peak_power = max([r["power_kw"] for r in result_data])
    peak_hour = next(r["hour"] for r in result_data if r["power_kw"] == peak_power)
    
    return {
        "status": "success",
        "location": f"{city} ({loc_info['lat']}, {loc_info['lon']})",
        "date": date,
        "system_config": {
            "capacity_kw": capacity_kw,
            "tilt": tilt,
            "azimuth": azimuth
        },
        "data_source": data_source,
        "summary": {
            "total_generation_kwh": round(total_energy, 2),
            "peak_power_kw": round(peak_power, 2),
            "peak_time": f"{peak_hour}:00",
            "equivalent_sun_hours": round(total_energy / capacity_kw, 2) if capacity_kw > 0 else 0
        },
        "hourly_curve": result_data
    }

# ==========================================
# 如何在 LangGraph 中使用此 Tool
# ==========================================
if __name__ == "__main__":
    # 模拟 LangGraph 中的调用
    from langchain_core.messages import HumanMessage
    from langgraph.prebuilt import create_react_agent
    from langchain_openai import ChatOpenAI
    
    # 注意：实际运行需要配置 OPENAI_API_KEY
    # llm = ChatOpenAI(model="gpt-4o")
    
    # 将工具注册到 Agent
    tools = [generate_pv_profile]
    
    # 创建简单的 Agent (React 模式)
    # agent_executor = create_react_agent(llm, tools)
    
    # 测试直接调用工具函数 (无需 LLM)
    print("=== 测试工具直接调用 ===")
    result = generate_pv_profile.invoke({
        "city": "北京",
        "date": "2024-06-21",
        "capacity_kw": 100,
        "tilt": 26
    })
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 输出示例结构:
    # {
    #   "status": "success",
    #   "location": "北京 (39.9042, 116.4074)",
    #   "summary": { "total_generation_kwh": 385.5, "peak_power_kw": 88.2, ... },
    #   "hourly_curve": [ {"hour": 0, "power_kw": 0}, ..., {"hour": 12, "power_kw": 88.2}, ... ]
    # }
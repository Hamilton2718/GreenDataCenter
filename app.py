from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import json
import time
import threading
import queue
from typing import Dict, Any, Generator

app = Flask(__name__)
CORS(app)  # 启用CORS，允许跨域请求

# 模拟多智能体编排框架
class AgentOrchestrator:
    """
    多智能体编排器
    """
    
    def __init__(self):
        self.agents = {
            "env_analysis": EnvironmentAnalysisAgent(),
            "energy_plan": EnergyPlanAgent(),
            "cooling_design": CoolingDesignAgent()
        }
    
    def process_task(self, task: Dict[str, Any], result_queue: queue.Queue):
        """
        处理任务
        """
        try:
            # 1. 环境分析Agent
            result_queue.put({
                "agent_id": "env_analysis",
                "status": "processing",
                "thought": "正在分析地理位置和环境数据..."
            })
            env_result = self.agents["env_analysis"].analyze(task)
            result_queue.put({
                "agent_id": "env_analysis",
                "status": "completed",
                "result": env_result,
                "thought": "环境数据分析完成"
            })
            time.sleep(1)
            
            # 2. 能源规划Agent
            result_queue.put({
                "agent_id": "energy_plan",
                "status": "processing",
                "thought": "正在制定能源规划方案..."
            })
            energy_result = self.agents["energy_plan"].plan(env_result)
            result_queue.put({
                "agent_id": "energy_plan",
                "status": "completed",
                "result": energy_result,
                "thought": "能源规划方案制定完成"
            })
            time.sleep(1)
            
            # 3. 制冷设计Agent
            result_queue.put({
                "agent_id": "cooling_design",
                "status": "processing",
                "thought": "正在设计制冷方案..."
            })
            cooling_result = self.agents["cooling_design"].design(env_result)
            result_queue.put({
                "agent_id": "cooling_design",
                "status": "completed",
                "result": cooling_result,
                "thought": "制冷方案设计完成"
            })
            time.sleep(1)
            
            # 4. 任务完成
            result_queue.put({
                "agent_id": "orchestrator",
                "status": "completed",
                "thought": "所有任务已完成"
            })
        except Exception as e:
            result_queue.put({
                "agent_id": "orchestrator",
                "status": "error",
                "thought": f"任务执行失败: {str(e)}"
            })
        finally:
            # 发送结束信号
            result_queue.put(None)

# 导入DataCenterAgent1
from nodes.requirement_analysis_node import DataCenterAgent1

# 环境分析Agent
class EnvironmentAnalysisAgent:
    """
    环境分析Agent
    """
    
    def __init__(self):
        """
        初始化环境分析Agent
        """
        self.agent1 = DataCenterAgent1()
    
    def analyze(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析环境数据
        """
        # 从task中获取参数
        location = task.get('location', '')
        business_type = task.get('business_type', '通用')
        planned_area = task.get('planned_area', 0)  # 平方米
        planned_load = task.get('planned_load', 0)  # kW
        Computing_power_density = task.get('Computing_power_density', 0)  # kW/机柜
        priority = task.get('priority', '环保型')
        green_energy_target = task.get('green_energy_target', 90)  # %
        pue_target = task.get('pue_target', 1.2)
        budget_constraint = task.get('budget_constraint', 0)  # 万元
        
        # 使用DataCenterAgent1获取环境数据
        environmental_data = self.agent1.get_environmental_data(location)
        
        # 使用DataCenterAgent1获取电价数据
        electricity_price = self.agent1.get_electricity_price(location)
        
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
            "timestamp": "2026-03-14"  # 使用当前时间
        }
        
        return standardized_data

# 导入energy_planner_node
from nodes.energy_planner_node import energy_planner_node

# 导入cooling_specialist_node
from nodes.cooling_specialist_node import cooling_specialist_node
from nodes.review_node import review_node
from nodes.financial_consultant_node import financial_consultant_node
from nodes.final_report_node import final_report_node

# 能源规划Agent
class EnergyPlanAgent:
    """
    能源规划Agent
    """
    
    def plan(self, env_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        制定能源规划方案
        """
        # 使用energy_planner_node生成能源规划方案
        result = energy_planner_node({
            "user_requirements": env_data.get("project_info", {}),
            "environmental_data": env_data.get("environmental_data", {}),
            "electricity_price": env_data.get("electricity_price", {})
        })
        
        # 提取能源规划方案
        energy_plan = result.get("energy_plan", {})
        
        # 构建返回数据
        return {
            "solar": {
                "ratio": 30,
                "capacity": 12
            },
            "storage": {
                "ratio": 15,
                "capacity": 6,
                "energy": 8
            },
            "ppa": {
                "ratio": 40,
                "capacity": 16
            },
            "grid": {
                "ratio": 15,
                "capacity": 6
            },
            "llm_report": energy_plan.get("llm_report", ""),
            "pv_capacity": energy_plan.get("pv_capacity", 0.0),
            "wind_capacity": energy_plan.get("wind_capacity", 0.0),
            "storage_capacity": energy_plan.get("storage_capacity", 0.0),
            "storage_power": energy_plan.get("storage_power", 0.0),
            "ppa_ratio": energy_plan.get("ppa_ratio", 0.0),
            "grid_ratio": energy_plan.get("grid_ratio", 0.0),
            "estimated_self_consumption": energy_plan.get("estimated_self_consumption", 0.0),
            "estimated_green_ratio": energy_plan.get("estimated_green_ratio", 0.0),
            "price_data_cn": energy_plan.get("price_data_cn", {}),
            "project_context": energy_plan.get("project_context", ""),
            "api_data": energy_plan.get("api_data", "")
        }

# 制冷设计Agent
class CoolingDesignAgent:
    """
    制冷设计Agent
    """
    
    def design(self, env_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        设计制冷方案
        """
        # 模拟制冷设计
        return {
            "primary": "间接蒸发冷却",
            "secondary": "液冷机柜",
            "pue": 1.18
        }

# 创建编排器实例
orchestrator = AgentOrchestrator()

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    分析项目需求
    """
    try:
        # 获取前端发送的JSON数据
        data = request.json
        print(f"接收到的前端数据: {data}")
        
        # 创建结果队列
        result_queue = queue.Queue()
        
        # 启动后台线程处理任务
        thread = threading.Thread(
            target=orchestrator.process_task,
            args=(data, result_queue)
        )
        thread.daemon = True
        thread.start()
        
        # 返回任务ID
        return jsonify({
            "task_id": "task_" + str(int(time.time())),
            "status": "started"
        })
    except Exception as e:
        print(f"错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream/analyze', methods=['POST'])
def stream_analyze():
    """
    流式返回分析结果
    """
    try:
        # 获取前端发送的JSON数据
        data = request.json
        print(f"接收到的前端数据: {data}")
        
        # 创建结果队列
        result_queue = queue.Queue()
        
        # 启动后台线程处理任务
        thread = threading.Thread(
            target=orchestrator.process_task,
            args=(data, result_queue)
        )
        thread.daemon = True
        thread.start()
        
        # 流式返回结果
        def generate():
            while True:
                try:
                    # 从队列中获取结果
                    result = result_queue.get(timeout=30)
                    
                    # 检查是否结束
                    if result is None:
                        break
                    
                    # 转换为NDJSON格式
                    yield json.dumps(result) + '\n'
                    
                    # 模拟延迟，使流式效果更明显
                    time.sleep(0.5)
                except queue.Empty:
                    break
        
        return Response(
            generate(),
            mimetype='application/x-ndjson'
        )
    except Exception as e:
        print(f"错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/agent2/energy-plan', methods=['POST'])
def agent2_energy_plan():
    """
    Agent 2专用接口：生成能源规划方案
    """
    try:
        # 获取前端发送的JSON数据
        data = request.json
        print(f"接收到的Agent 2数据: {data}")
        
        # 调用energy_planner_node生成能源规划方案
        result = energy_planner_node(data)
        
        # 提取能源规划方案
        energy_plan = result.get("energy_plan", {})
        
        # 返回结果
        return jsonify({
            'success': True,
            'data': energy_plan
        })
    except Exception as e:
        print(f"错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/agent3/cooling-plan', methods=['POST'])
def agent3_cooling_plan():
    """
    Agent 3专用接口：生成制冷方案
    """
    try:
        # 获取前端发送的JSON数据
        data = request.json
        print(f"接收到的Agent 3数据: {data}")
        
        # 调用cooling_specialist_node生成制冷方案
        result = cooling_specialist_node(data)
        
        # 提取制冷方案
        cooling_plan = result.get("cooling_plan", {})
        
        # 返回结果
        return jsonify({
            'success': True,
            'data': cooling_plan
        })
    except Exception as e:
        print(f"错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/agent4/review', methods=['POST'])
def agent4_review():
    """
    Agent 4专用接口：方案验证与评估
    """
    try:
        # 获取前端发送的JSON数据
        data = request.json
        print(f"接收到的Agent 4数据: {data}")
        
        # 调用review_node进行方案评估
        result = review_node(data)
        
        # 提取评估结果和反馈信息
        review_result = result.get("review_result", {})
        feedback = result.get("feedback", {})
        
        # 返回结果
        return jsonify({
            'success': True,
            'data': {
                'review_result': review_result,
                'feedback': feedback,
                'iteration_count': result.get('iteration_count', 0)
            }
        })
    except Exception as e:
        print(f"错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/agent5/financial', methods=['POST'])
def agent5_financial():
    """
    Agent 5专用接口：综合评价与投资决策
    """
    try:
        # 获取前端发送的JSON数据
        data = request.json
        print(f"接收到的Agent 5数据: {data}")
        
        # 调用financial_consultant_node进行财务分析
        result = financial_consultant_node(data)
        
        # 提取财务分析结果
        financial_analysis = result.get("financial_analysis", {})
        
        # 返回结果
        return jsonify({
            'success': True,
            'data': {
                'financial_analysis': financial_analysis
            }
        })
    except Exception as e:
        print(f"错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    """
    生成最终报告
    """
    try:
        # 获取前端发送的JSON数据
        data = request.json
        print(f"接收到的报告生成数据: {data}")
        
        # 调用final_report_node生成最终报告
        result = final_report_node(data)
        
        # 返回完整的状态数据
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        print(f"错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

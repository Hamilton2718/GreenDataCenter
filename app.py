from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import json
import time
import threading
import queue
from typing import Dict, Any, Generator
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class AgentOrchestrator:
    def __init__(self):
        self.agents = {
            "env_analysis": EnvironmentAnalysisAgent(),
            "energy_plan": EnergyPlanAgent(),
            "cooling_design": CoolingDesignAgent()
        }
    
    def process_task(self, task: Dict[str, Any], result_queue: queue.Queue):
        try:
            logger.info("=" * 60)
            logger.info("🚀 [AgentOrchestrator] 开始处理任务")
            logger.info("=" * 60)
            
            logger.info("📋 [Agent 1: 环境分析] 开始处理...")
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
            logger.info("✅ [Agent 1: 环境分析] 处理完成")
            time.sleep(1)
            
            logger.info("⚡ [Agent 2: 能源规划] 开始处理...")
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
            logger.info("✅ [Agent 2: 能源规划] 处理完成")
            time.sleep(1)
            
            logger.info("❄️ [Agent 3: 制冷设计] 开始处理...")
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
            logger.info("✅ [Agent 3: 制冷设计] 处理完成")
            time.sleep(1)
            
            logger.info("🎉 [AgentOrchestrator] 所有任务已完成")
            result_queue.put({
                "agent_id": "orchestrator",
                "status": "completed",
                "thought": "所有任务已完成"
            })
        except Exception as e:
            logger.error(f"❌ [AgentOrchestrator] 任务执行失败: {str(e)}", exc_info=True)
            result_queue.put({
                "agent_id": "orchestrator",
                "status": "error",
                "thought": f"任务执行失败: {str(e)}"
            })
        finally:
            result_queue.put(None)

from nodes.requirement_analysis_node import DataCenterAgent1

class EnvironmentAnalysisAgent:
    def __init__(self):
        self.agent1 = DataCenterAgent1()
    
    def analyze(self, task: Dict[str, Any]) -> Dict[str, Any]:
        location = task.get('location', '')
        business_type = task.get('business_type', '通用')
        planned_area = task.get('planned_area', 0)
        planned_load = task.get('planned_load', 0)
        Computing_power_density = task.get('Computing_power_density', 0)
        priority = task.get('priority', '环保型')
        green_energy_target = task.get('green_energy_target', 90)
        pue_target = task.get('pue_target', 1.2)
        budget_constraint = task.get('budget_constraint', 0)
        
        logger.info(f"📍 位置: {location}")
        logger.info(f"🏢 业务类型: {business_type}")
        logger.info(f"⚡ 计划负荷: {planned_load} kW")
        logger.info(f"🌿 绿电目标: {green_energy_target}%")
        
        logger.info("🌐 正在获取环境数据...")
        environmental_data = self.agent1.get_environmental_data(location)
        
        logger.info("💰 正在获取电价数据...")
        electricity_price = self.agent1.get_electricity_price(location)
        
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
            "timestamp": "2026-03-14"
        }
        
        return standardized_data

from nodes.energy_planner_node import energy_planner_node
from nodes.cooling_specialist_node import cooling_specialist_node
from nodes.review_node import review_node
from nodes.financial_consultant_node import financial_consultant_node
from nodes.final_report_node import final_report_node

class EnergyPlanAgent:
    def plan(self, env_data: Dict[str, Any]) -> Dict[str, Any]:
        result = energy_planner_node({
            "user_requirements": env_data.get("project_info", {}),
            "environmental_data": env_data.get("environmental_data", {}),
            "electricity_price": env_data.get("electricity_price", {})
        })
        
        energy_plan = result.get("energy_plan", {})
        
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

class CoolingDesignAgent:
    def design(self, env_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "primary": "间接蒸发冷却",
            "secondary": "液冷机柜",
            "pue": 1.18
        }

orchestrator = AgentOrchestrator()

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        logger.info(f"📥 接收到前端数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
        
        result_queue = queue.Queue()
        
        thread = threading.Thread(
            target=orchestrator.process_task,
            args=(data, result_queue)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "task_id": "task_" + str(int(time.time())),
            "status": "started"
        })
    except Exception as e:
        logger.error(f"❌ 错误: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream/analyze', methods=['POST'])
def stream_analyze():
    try:
        data = request.json
        logger.info(f"📥 接收到前端数据 (流式): {json.dumps(data, ensure_ascii=False, indent=2)}")
        
        result_queue = queue.Queue()
        
        thread = threading.Thread(
            target=orchestrator.process_task,
            args=(data, result_queue)
        )
        thread.daemon = True
        thread.start()
        
        def generate():
            while True:
                try:
                    result = result_queue.get(timeout=30)
                    
                    if result is None:
                        break
                    
                    logger.info(f"📤 发送流式数据: {result.get('agent_id', 'unknown')} - {result.get('status', 'unknown')}")
                    yield json.dumps(result) + '\n'
                    
                    time.sleep(0.5)
                except queue.Empty:
                    break
        
        return Response(
            generate(),
            mimetype='application/x-ndjson'
        )
    except Exception as e:
        logger.error(f"❌ 错误: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/agent2/energy-plan', methods=['POST'])
def agent2_energy_plan():
    try:
        data = request.json
        logger.info("=" * 60)
        logger.info("⚡ [API] Agent 2 能源规划请求")
        logger.info("=" * 60)
        logger.info(f"📥 请求数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
        
        result = energy_planner_node(data)
        
        energy_plan = result.get("energy_plan", {})
        
        logger.info("✅ [API] Agent 2 能源规划完成")
        logger.info(f"📊 光伏容量: {energy_plan.get('pv_capacity', 0)} kW")
        logger.info(f"📊 储能容量: {energy_plan.get('storage_capacity', 0)} kWh")
        logger.info(f"📊 绿电占比: {energy_plan.get('estimated_green_ratio', 0)}%")
        
        return jsonify({
            'success': True,
            'data': energy_plan
        })
    except Exception as e:
        logger.error(f"❌ [API] Agent 2 错误: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/agent3/cooling-plan', methods=['POST'])
def agent3_cooling_plan():
    try:
        data = request.json
        logger.info("=" * 60)
        logger.info("❄️ [API] Agent 3 制冷方案请求")
        logger.info("=" * 60)
        logger.info(f"📥 请求数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
        
        result = cooling_specialist_node(data)
        
        cooling_plan = result.get("cooling_plan", {})
        
        logger.info("✅ [API] Agent 3 制冷方案完成")
        logger.info(f"📊 制冷技术: {cooling_plan.get('cooling_technology', 'N/A')}")
        logger.info(f"📊 预计 PUE: {cooling_plan.get('estimated_pue', 'N/A')}")
        logger.info(f"📊 预计 WUE: {cooling_plan.get('predicted_wue', 'N/A')}")
        
        return jsonify({
            'success': True,
            'data': cooling_plan
        })
    except Exception as e:
        logger.error(f"❌ [API] Agent 3 错误: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/agent4/review', methods=['POST'])
def agent4_review():
    try:
        data = request.json
        logger.info("=" * 60)
        logger.info("🔍 [API] Agent 4 方案验证请求")
        logger.info("=" * 60)
        logger.info(f"📥 请求数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
        
        result = review_node(data)
        
        review_result = result.get("review_result", {})
        feedback = result.get("feedback", {})
        
        logger.info("✅ [API] Agent 4 方案验证完成")
        logger.info(f"📊 评估结果: {'通过' if review_result.get('passed') else '不通过'}")
        logger.info(f"📊 评估工具: {review_result.get('evaluator', 'Unknown')}")
        
        return jsonify({
            'success': True,
            'data': {
                'review_result': review_result,
                'feedback': feedback,
                'iteration_count': result.get('iteration_count', 0)
            }
        })
    except Exception as e:
        logger.error(f"❌ [API] Agent 4 错误: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/agent5/financial', methods=['POST'])
def agent5_financial():
    try:
        data = request.json
        logger.info("=" * 60)
        logger.info("💰 [API] Agent 5 财务分析请求")
        logger.info("=" * 60)
        logger.info(f"📥 请求数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
        
        result = financial_consultant_node(data)
        
        financial_analysis = result.get("financial_analysis", {})
        
        logger.info("✅ [API] Agent 5 财务分析完成")
        logger.info(f"📊 总投资: {financial_analysis.get('capex_total', 0)} 万元")
        logger.info(f"📊 投资回收期: {financial_analysis.get('payback_years', 0)} 年")
        
        return jsonify({
            'success': True,
            'data': {
                'financial_analysis': financial_analysis
            }
        })
    except Exception as e:
        logger.error(f"❌ [API] Agent 5 错误: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    try:
        data = request.json
        logger.info("=" * 60)
        logger.info("📄 [API] 报告生成请求")
        logger.info("=" * 60)
        logger.info(f"📥 请求数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
        
        result = final_report_node(data)
        
        logger.info("✅ [API] 报告生成完成")
        
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        logger.error(f"❌ [API] 报告生成错误: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 GreenDataCenter 后端服务启动")
    logger.info("=" * 60)
    logger.info("🌐 服务地址: http://localhost:5001")
    logger.info("📝 日志级别: INFO")
    app.run(host='0.0.0.0', port=5001, debug=True)

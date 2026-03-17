<template>
  <div class="requirement-container">
    <el-card class="section-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <el-icon><Location /></el-icon>
          <span>项目基础信息</span>
        </div>
      </template>
      
      <el-form :model="form" label-width="120px" :rules="rules" ref="formRef">
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="地理位置" prop="location">
              <el-input v-model="form.location" placeholder="例如：乌兰察布">
                <template #append>
                  <el-button @click="showMap = true">
                    <el-icon><Position /></el-icon> 选择
                  </el-button>
                </template>
              </el-input>
            </el-form-item>
          </el-col>
          
          <el-col :span="12">
            <el-form-item label="业务类型" prop="businessType">
              <el-select v-model="form.businessType" placeholder="请选择">
                <el-option label="大模型训练（高恒定负荷）" value="training" />
                <el-option label="云存储（低密度）" value="storage" />
                <el-option label="边缘计算" value="edge" />
                <el-option label="混合业务" value="mixed" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="20">
          <el-col :span="8">
            <el-form-item label="计划面积(m²)" prop="area">
              <el-input-number v-model="form.area" :min="100" :max="100000" />
            </el-form-item>
          </el-col>
          
          <el-col :span="8">
            <el-form-item label="计划负荷(MW)" prop="load">
              <el-input-number v-model="form.load" :min="1" :max="500" :step="1" />
            </el-form-item>
          </el-col>
          
          <el-col :span="8">
            <el-form-item label="算力密度" prop="density">
              <el-radio-group v-model="form.density">
                <el-radio-button :value="8">8kW/机柜 (风冷)</el-radio-button>
                <el-radio-button :value="30">30kW/机柜 (液冷)</el-radio-button>
              </el-radio-group>
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>
    </el-card>

    <el-card class="section-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <el-icon><Flag /></el-icon>
          <span>项目目标设定</span>
        </div>
      </template>

      <el-form label-width="120px">
        <el-row :gutter="20">
          <el-col :span="24">
            <el-form-item label="优先级">
              <el-checkbox-group v-model="form.priority">
                <el-checkbox value="reliable">可靠型</el-checkbox>
                <el-checkbox value="economic">经济型</el-checkbox>
                <el-checkbox value="green">环保型</el-checkbox>
              </el-checkbox-group>
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="绿电目标">
              <el-slider 
                v-model="form.greenTarget" 
                :min="0" 
                :max="100"
                :format-tooltip="(val: number) => val + '%'"
              />
              <span class="target-value">{{ form.greenTarget }}%</span>
            </el-form-item>
          </el-col>
          
          <el-col :span="12">
            <el-form-item label="PUE目标">
              <el-slider 
                v-model="form.pueTarget" 
                :min="1.0" 
                :max="2.0"
                :step="0.05"
                :format-tooltip="(val: number) => val.toFixed(2)"
              />
              <span class="target-value">{{ form.pueTarget.toFixed(2) }}</span>
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="预算约束(万元)">
              <el-input-number v-model="form.budget" :min="0" :step="1000" />
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>
    </el-card>

    <el-card class="section-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <el-icon><Cpu /></el-icon>
          <span>智能分析状态</span>
        </div>
      </template>

      <div class="agent-status-container">
        <el-steps :active="activeStep" finish-status="success" align-center>
          <el-step title="环境分析" :description="agentStatus.env_analysis.thought">
            <template #icon>
              <el-icon :color="getAgentColor('env_analysis')">
                <component :is="getAgentIcon('env_analysis')" />
              </el-icon>
            </template>
          </el-step>
          <el-step title="能源规划" :description="agentStatus.energy_plan.thought">
            <template #icon>
              <el-icon :color="getAgentColor('energy_plan')">
                <component :is="getAgentIcon('energy_plan')" />
              </el-icon>
            </template>
          </el-step>
          <el-step title="制冷设计" :description="agentStatus.cooling_design.thought">
            <template #icon>
              <el-icon :color="getAgentColor('cooling_design')">
                <component :is="getAgentIcon('cooling_design')" />
              </el-icon>
            </template>
          </el-step>
        </el-steps>
      </div>
    </el-card>

    <el-card class="section-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <el-icon><DataAnalysis /></el-icon>
          <span>环境数据加载</span>
        </div>
      </template>

      <el-row :gutter="20">
        <el-col :span="8">
          <el-statistic title="年均温度" :value="envData.climate.avgTemp" :precision="2" suffix="°C" />
        </el-col>
        <el-col :span="8">
          <el-statistic title="年均风速" :value="envData.climate.windSpeed" :precision="2" suffix="m/s" />
        </el-col>
        <el-col :span="8">
          <el-statistic title="年日照时数" :value="envData.climate.solarRadiation" :precision="2" suffix="h" />
        </el-col>
      </el-row>

      <el-divider />

      <el-row :gutter="20">
        <el-col :span="8">
          <el-statistic title="尖峰电价" :value="envData.electricity.peakPrice" :precision="4" suffix="元/kWh" />
        </el-col>
        <el-col :span="8">
          <el-statistic title="高峰电价" :value="envData.electricity.highPrice" :precision="4" suffix="元/kWh" />
        </el-col>
        <el-col :span="8">
          <el-statistic title="平段电价" :value="envData.electricity.flatPrice" :precision="4" suffix="元/kWh" />
        </el-col>
      </el-row>
      <el-row :gutter="20" style="margin-top: 20px">
        <el-col :span="8">
          <el-statistic title="低谷电价" :value="envData.electricity.valleyPrice" :precision="4" suffix="元/kWh" />
        </el-col>
        <el-col :span="8">
          <el-statistic title="深谷电价" :value="envData.electricity.deepValleyPrice" :precision="4" suffix="元/kWh" />
        </el-col>
        <el-col :span="8">
          <el-statistic title="碳排因子" :value="envData.carbonFactor" :precision="4" suffix="kg CO₂/kWh" />
        </el-col>
      </el-row>
      <el-row :gutter="20" style="margin-top: 20px">
        <el-col :span="8">
          <el-statistic title="最大峰谷价差" :value="envData.electricity.maxPriceDiff" :precision="4" suffix="元/kWh" />
        </el-col>
      </el-row>

      <div class="data-refresh">
        <el-button type="success" plain @click="refreshData">
          <el-icon><Refresh /></el-icon> 刷新数据
        </el-button>
        <span class="refresh-time">最后更新：{{ refreshTime }}</span>
      </div>
    </el-card>

    <div class="action-buttons">
      <el-button size="large" @click="saveDraft">
        <el-icon><Document /></el-icon> 保存草稿
      </el-button>
      <el-button type="primary" size="large" @click="startDesign">
        <el-icon><ArrowRight /></el-icon> 开始规划设计
      </el-button>
    </div>

    <!-- 地图选择对话框 -->
    <el-dialog v-model="showMap" title="选择地理位置" width="70%">
      <div style="height: 400px; background: #f5f7fa; display: flex; align-items: center; justify-content: center">
        <el-empty description="地图组件预留位置（可集成高德/百度地图）" />
      </div>
      <template #footer>
        <el-button @click="showMap = false">取消</el-button>
        <el-button type="primary" @click="selectLocation">确认选择</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useProjectStore } from '../stores/projectStore'

const router = useRouter()
const projectStore = useProjectStore()

const form = reactive({
  location: projectStore.requirement.location,
  businessType: projectStore.requirement.businessType,
  area: projectStore.requirement.area,
  load: projectStore.requirement.load,
  density: projectStore.requirement.density,
  priority: projectStore.requirement.priority,
  greenTarget: projectStore.requirement.greenTarget,
  pueTarget: projectStore.requirement.pueTarget,
  budget: projectStore.requirement.budget
})

const envData = reactive(projectStore.envData)
const showMap = ref(false)
const formRef = ref()
const refreshTime = ref(new Date().toLocaleString())

const rules = {
  location: [{ required: true, message: '请选择地理位置', trigger: 'blur' }],
  businessType: [{ required: true, message: '请选择业务类型', trigger: 'change' }],
  area: [{ required: true, message: '请输入计划面积', trigger: 'blur' }],
  load: [{ required: true, message: '请输入计划负荷', trigger: 'blur' }]
}

const isLoading = ref(false)

interface AgentStatusItem {
  status: string;
  thought: string;
}

interface AgentStatusMap {
  env_analysis: AgentStatusItem;
  energy_plan: AgentStatusItem;
  cooling_design: AgentStatusItem;
  [key: string]: AgentStatusItem;
}

const agentStatus = ref<AgentStatusMap>({
  env_analysis: { status: 'idle', thought: '' },
  energy_plan: { status: 'idle', thought: '' },
  cooling_design: { status: 'idle', thought: '' }
})

const refreshData = async () => {
  try {
    ElMessage.info('正在从后端获取数据...')
    isLoading.value = true
    
    // 重置Agent状态
    Object.keys(agentStatus.value).forEach(key => {
      agentStatus.value[key] = { status: 'idle', thought: '' }
    })
    
    // 构建后端期望的数据格式
    const backendData = {
      location: form.location,
      business_type: form.businessType,
      planned_area: form.area,
      planned_load: form.load * 1000,  // 转换为kW
      Computing_power_density: form.density,
      priority: form.priority.join(','),
      green_energy_target: form.greenTarget,
      pue_target: form.pueTarget,
      budget_constraint: form.budget
    }
    
    // 发起流式请求
    const response = await fetch('/api/stream/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(backendData)
    })
    
    if (!response.ok) {
      throw new Error('API调用失败')
    }
    
    // 处理流式响应
    if (!response.body) {
      throw new Error('响应体为空')
    }
    
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    
    while (true) {
      const { done, value } = await reader.read()
      
      if (done) {
        break
      }
      
      // 检查value是否为null
      if (!value) {
        continue
      }
      
      // 解码数据
      const chunk = decoder.decode(value, { stream: true })
      
      // 处理NDJSON格式的数据
      const lines = chunk.split('\n').filter(line => line.trim())
      for (const line of lines) {
        try {
          const data = JSON.parse(line)
          console.log('接收到的流式数据:', data)
          
          // 更新Agent状态
          if (data.agent_id in agentStatus.value) {
            agentStatus.value[data.agent_id] = {
              status: data.status,
              thought: data.thought
            }
          }
          
          // 处理环境数据
          if (data.agent_id === 'env_analysis' && data.status === 'completed' && data.result) {
            const result = data.result
            console.log('环境分析结果:', result)
            
            // 检查数据是否存在
            if (result.environmental_data) {
              console.log('环境数据:', result.environmental_data)
              envData.climate.avgTemp = result.environmental_data.annual_temperature || 0
              envData.climate.windSpeed = result.environmental_data.annual_wind_speed || 0
              envData.climate.solarRadiation = result.environmental_data.annual_sunshine_hours || 0
              envData.carbonFactor = result.environmental_data.carbon_emission_factor || 0
              console.log('碳排因子:', envData.carbonFactor)
            }
            
            if (result.electricity_price) {
              console.log('电价数据:', result.electricity_price)
              envData.electricity.peakPrice = result.electricity_price.尖峰电价 || 0
              envData.electricity.highPrice = result.electricity_price.高峰电价 || 0
              envData.electricity.flatPrice = result.electricity_price.平段电价 || 0
              envData.electricity.valleyPrice = result.electricity_price.低谷电价 || 0
              envData.electricity.deepValleyPrice = result.electricity_price.深谷电价 || 0
              envData.electricity.maxPriceDiff = result.electricity_price.最大峰谷价差 || 0
              console.log('电价数据更新后:', envData.electricity)
            }
            
            console.log('更新后的envData:', envData)
            
            // 直接更新projectStore中的数据，确保响应性
            projectStore.updateEnvData({
              climate: {
                avgTemp: envData.climate.avgTemp,
                windSpeed: envData.climate.windSpeed,
                solarRadiation: envData.climate.solarRadiation
              },
              electricity: {
                peakPrice: envData.electricity.peakPrice,
                highPrice: envData.electricity.highPrice,
                flatPrice: envData.electricity.flatPrice,
                valleyPrice: envData.electricity.valleyPrice,
                deepValleyPrice: envData.electricity.deepValleyPrice,
                maxPriceDiff: envData.electricity.maxPriceDiff
              },
              carbonFactor: envData.carbonFactor
            })
          }
          
          // 处理能源规划数据
          if (data.agent_id === 'energy_plan' && data.status === 'completed' && data.result) {
            const result = data.result
            console.log('能源规划结果:', result)
            
            // 直接更新projectStore中的数据，确保响应性
            projectStore.updateEnergyPlan(result)
            console.log('更新后的energyPlan:', projectStore.energyPlan)
          }
        } catch (error) {
          console.error('解析流式数据失败:', error)
        }
      }
    }
    
    // 更新刷新时间
    refreshTime.value = new Date().toLocaleString()
    
    ElMessage.success('数据已从后端刷新')
  } catch (error) {
    console.error('获取数据失败:', error)
    ElMessage.error('获取数据失败，请稍后重试')
  } finally {
    isLoading.value = false
  }
}

const saveDraft = () => {
  projectStore.updateRequirement(form)
  ElMessage.success('草稿保存成功')
}

const startDesign = async () => {
  await formRef.value?.validate()
  
  try {
    ElMessage.info('正在从后端获取数据...')
    isLoading.value = true
    
    // 重置Agent状态
    Object.keys(agentStatus.value).forEach(key => {
      agentStatus.value[key] = { status: 'idle', thought: '' }
    })
    
    // 构建后端期望的数据格式
    const backendData = {
      location: form.location,
      business_type: form.businessType,
      planned_area: form.area,
      planned_load: form.load * 1000,  // 转换为kW
      Computing_power_density: form.density,
      priority: form.priority.join(','),
      green_energy_target: form.greenTarget,
      pue_target: form.pueTarget,
      budget_constraint: form.budget
    }
    
    // 发起流式请求
    const response = await fetch('/api/stream/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(backendData)
    })
    
    if (!response.ok) {
      throw new Error('API调用失败')
    }
    
    // 处理流式响应
    if (!response.body) {
      throw new Error('响应体为空')
    }
    
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    
    while (true) {
      const { done, value } = await reader.read()
      
      if (done) {
        break
      }
      
      // 检查value是否为null
      if (!value) {
        continue
      }
      
      // 解码数据
      const chunk = decoder.decode(value, { stream: true })
      
      // 处理NDJSON格式的数据
      const lines = chunk.split('\n').filter(line => line.trim())
      for (const line of lines) {
        try {
          const data = JSON.parse(line)
          console.log('接收到的流式数据:', data)
          
          // 更新Agent状态
          if (data.agent_id in agentStatus.value) {
            agentStatus.value[data.agent_id] = {
              status: data.status,
              thought: data.thought
            }
          }
          
          // 处理环境数据
          if (data.agent_id === 'env_analysis' && data.status === 'completed' && data.result) {
            const result = data.result
            console.log('环境分析结果:', result)
            
            // 检查数据是否存在
            if (result.environmental_data) {
              console.log('环境数据:', result.environmental_data)
              envData.climate.avgTemp = result.environmental_data.annual_temperature || 0
              envData.climate.windSpeed = result.environmental_data.annual_wind_speed || 0
              envData.climate.solarRadiation = result.environmental_data.annual_sunshine_hours || 0
              envData.carbonFactor = result.environmental_data.carbon_emission_factor || 0
              console.log('碳排因子:', envData.carbonFactor)
            }
            
            if (result.electricity_price) {
              console.log('电价数据:', result.electricity_price)
              envData.electricity.peakPrice = result.electricity_price.尖峰电价 || 0
              envData.electricity.highPrice = result.electricity_price.高峰电价 || 0
              envData.electricity.flatPrice = result.electricity_price.平段电价 || 0
              envData.electricity.valleyPrice = result.electricity_price.低谷电价 || 0
              envData.electricity.deepValleyPrice = result.electricity_price.深谷电价 || 0
              envData.electricity.maxPriceDiff = result.electricity_price.最大峰谷价差 || 0
              console.log('电价数据更新后:', envData.electricity)
            }
            
            console.log('更新后的envData:', envData)
            
            // 直接更新projectStore中的数据，确保响应性
            projectStore.updateEnvData({
              climate: {
                avgTemp: envData.climate.avgTemp,
                windSpeed: envData.climate.windSpeed,
                solarRadiation: envData.climate.solarRadiation
              },
              electricity: {
                peakPrice: envData.electricity.peakPrice,
                highPrice: envData.electricity.highPrice,
                flatPrice: envData.electricity.flatPrice,
                valleyPrice: envData.electricity.valleyPrice,
                deepValleyPrice: envData.electricity.deepValleyPrice,
                maxPriceDiff: envData.electricity.maxPriceDiff
              },
              carbonFactor: envData.carbonFactor
            })
          }
          
          // 所有任务完成
          if (data.agent_id === 'orchestrator' && data.status === 'completed') {
            // 更新projectStore
            projectStore.updateRequirement(form)
            projectStore.updateAgentStatus('agent1', true)
            
            // 更新刷新时间
            refreshTime.value = new Date().toLocaleString()
            
            // 跳转到设计页面
            setTimeout(() => {
              router.push('/design')
              ElMessage.success('需求解析完成，进入方案设计阶段')
            }, 1000)
          }
        } catch (error) {
          console.error('解析流式数据失败:', error)
        }
      }
    }
  } catch (error) {
    console.error('获取数据失败:', error)
    ElMessage.error('获取数据失败，请稍后重试')
  } finally {
    isLoading.value = false
  }
}

const activeStep = computed(() => {
  if (agentStatus.value.env_analysis.status === 'completed') {
    if (agentStatus.value.energy_plan.status === 'completed') {
      if (agentStatus.value.cooling_design.status === 'completed') {
        return 2
      }
      return 1
    }
    return 0
  }
  return -1
})

const getAgentColor = (agentId: string) => {
  const agentItem = agentStatus.value[agentId as keyof typeof agentStatus.value]
  const status = agentItem?.status || 'idle'
  switch (status) {
    case 'processing':
      return '#409eff'
    case 'completed':
      return '#67c23a'
    case 'error':
      return '#f56c6c'
    default:
      return '#909399'
  }
}

const getAgentIcon = (agentId: string) => {
  const agentItem = agentStatus.value[agentId as keyof typeof agentStatus.value]
  const status = agentItem?.status || 'idle'
  switch (status) {
    case 'processing':
      return 'Loading'
    case 'completed':
      return 'Check'
    case 'error':
      return 'Close'
    default:
      return 'InfoFilled'
  }
}

const selectLocation = () => {
  showMap.value = false
  ElMessage.success('已选择：乌兰察布市集宁区')
}
</script>

<style scoped>
.requirement-container {
  max-width: 1200px;
  margin: 0 auto;
}

.section-card {
  margin-bottom: 24px;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: bold;
}

.target-value {
  margin-left: 16px;
  color: #409eff;
  font-weight: bold;
}

.data-refresh {
  margin-top: 24px;
  display: flex;
  align-items: center;
  gap: 16px;
}

.refresh-time {
  color: #909399;
  font-size: 0.9rem;
}

.agent-status-container {
  padding: 20px 0;
}

.el-steps {
  margin-bottom: 20px;
}

.el-step__description {
  font-size: 12px;
  color: #606266;
  margin-top: 8px;
  max-width: 300px;
  text-align: center;
}

.el-step__icon {
  width: 36px;
  height: 36px;
  line-height: 36px;
  font-size: 18px;
}

.action-buttons {
  display: flex;
  justify-content: flex-end;
  gap: 16px;
  margin-top: 32px;
}
</style>
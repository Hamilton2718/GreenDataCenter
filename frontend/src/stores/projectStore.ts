import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export interface ProjectRequirement {
  location: string
  businessType: 'training' | 'storage' | 'edge' | 'mixed'
  area: number
  load: number
  density: 8 | 30
  priority: ('reliable' | 'economic' | 'green')[]
  greenTarget: number
  pueTarget: number
  budget?: number
}

export interface AgentStatus {
  agent1: boolean
  agent2: boolean
  agent3: boolean
  agent4: boolean
  agent5: boolean
}

export interface CoolingPlan {
  primary?: string
  secondary?: string
  pue?: number
  cooling_technology?: string
  estimated_pue?: number
  predicted_wue?: number
  scheme_detail_brief?: string
  waste_heat_recovery_strategy?: string
  cooling_project_info?: {
    location?: string
    it_load_kW?: number
    cabinet_power_kW?: number
    target_pue?: number
    green_energy_target?: number
    province?: string
  }
  cooling_calc_params?: {
    PUE_Limit?: number
    WUE_Limit?: number
    cabinet_power_limit?: number
    cooling_eff_coeff?: number
    facility_loss_coeff?: number
    regional_cooling_preference?: string
    waste_heat_recovery_coeff?: number
  }
  cooling_kpis?: {
    PUE_Limit?: number
    WUE_Limit?: number
    cooling_power_kw?: number
    corrected_cop?: number
    facility_loss_kw?: number
    predicted_PUE?: number
    predicted_WUE?: number
    waste_heat_recovery_kw?: number
  }
}

export const useProjectStore = defineStore('project', () => {
  // 项目需求
  const requirement = ref<ProjectRequirement>({
    location: '乌兰察布',
    businessType: 'training',
    area: 5000,
    load: 40,
    density: 30,
    priority: ['green'],
    greenTarget: 90,
    pueTarget: 1.2,
    budget: 30000
  })

  // Agent状态
  const agentStatus = ref<AgentStatus>({
    agent1: false,
    agent2: false,
    agent3: false,
    agent4: false,
    agent5: false
  })

  // 环境数据
  const envData = ref({
    climate: {
      avgTemp: 8.2,
      windSpeed: 5.6,
      solarRadiation: 1650
    },
    electricity: {
      peakPrice: 1.2,
      highPrice: 1.0,
      flatPrice: 0.8,
      valleyPrice: 0.4,
      deepValleyPrice: 0.3,
      maxPriceDiff: 0.9
    },
    carbonFactor: 0.581
  })

  // 能源方案
  const energyPlan = ref({
    solar: { ratio: 30, capacity: 12 },
    storage: { ratio: 15, capacity: 6, energy: 8 },
    ppa: { ratio: 40, capacity: 16 },
    grid: { ratio: 15, capacity: 6 },
    llm_report: '',
    pv_capacity: 0.0,
    wind_capacity: 0.0,
    storage_capacity: 0.0,
    storage_power: 0.0,
    ppa_ratio: 0.0,
    grid_ratio: 0.0,
    estimated_self_consumption: 0.0,
    estimated_green_ratio: 0.0,
    price_data_cn: {},
    project_context: '',
    api_data: ''
  })

  // 制冷方案
  const coolingPlan = ref<CoolingPlan>({
    primary: '间接蒸发冷却',
    secondary: '液冷机柜',
    pue: 1.18
  })

  // 仿真结果
  const simulationResult = ref({
    greenRatio: 92,
    pue: 1.18,
    storageEfficiency: 65,
    warnings: ['储能利用率可优化']
  })

  // 投资结果
  const investmentResult = ref({
    totalInvestment: 28000,
    annualCarbonSave: 12,
    paybackYears: 6.5
  })

  // 评估结果
  const reviewResult = ref({
    evaluation_text: '',
    passed: false,
    score: 0,
    evaluator: 'Unknown'
  })

  // 反馈信息
  const feedback = ref({
    passed: false,
    issues: [],
    suggestions: [],
    full_evaluation: '',
    retry_required: false,
    max_iterations_reached: false,
    iteration_count: 0
  })

  // 财务分析结果
  const financialAnalysis = ref({
    location: '',
    planned_load: 0,
    total_electricity: 0,
    annual_grid_purchase: 0,
    green_consumption: 0,
    ppa_volume: 0,
    green_ratio: 0,
    green_target: 0,
    actual_pue: 0,
    pue_target: 0,
    grid_price: 0,
    ppa_price: 0,
    carbon_price: 0,
    grid_cost: 0,
    ppa_cost: 0,
    pv_saving: 0,
    carbon_benefit: 0,
    carbon_compensation_cost: 0,
    total_cost: 0,
    capex_total: 0,
    annual_saving: 0,
    payback_years: 0,
    emission_reduction: 0,
    lifetime_reduction: 0,
    cooling_tech: '',
    curtailment_rate: 0,
    simulation_used: false,
    report_md: '',
    capex_breakdown: {
      pv_system: 0,
      storage_system: 0,
      cooling_system: 0
    }
  })

  // 最终报告
  const finalReport = ref('')

  // 计算属性
  const progress = computed(() => {
    const completed = Object.values(agentStatus.value).filter(v => v).length
    return (completed / 5) * 100
  })

  // 方法
  function updateRequirement(data: Partial<ProjectRequirement>) {
    requirement.value = { ...requirement.value, ...data }
  }

  function updateAgentStatus(agent: keyof AgentStatus, status: boolean) {
    agentStatus.value[agent] = status
  }

  function resetAll() {
    agentStatus.value = {
      agent1: false,
      agent2: false,
      agent3: false,
      agent4: false,
      agent5: false
    }
  }

  function updateEnvData(data: any) {
    envData.value = { ...envData.value, ...data }
  }

  function updateEnergyPlan(data: any) {
    energyPlan.value = { 
      ...energyPlan.value, 
      ...data,
      // 确保所有新字段都被正确更新
      pv_capacity: data.pv_capacity || energyPlan.value.pv_capacity,
      wind_capacity: data.wind_capacity || energyPlan.value.wind_capacity,
      storage_capacity: data.storage_capacity || energyPlan.value.storage_capacity,
      storage_power: data.storage_power || energyPlan.value.storage_power,
      ppa_ratio: data.ppa_ratio || energyPlan.value.ppa_ratio,
      grid_ratio: data.grid_ratio || energyPlan.value.grid_ratio,
      estimated_self_consumption: data.estimated_self_consumption || energyPlan.value.estimated_self_consumption,
      estimated_green_ratio: data.estimated_green_ratio || energyPlan.value.estimated_green_ratio,
      price_data_cn: data.price_data_cn || energyPlan.value.price_data_cn,
      project_context: data.project_context || energyPlan.value.project_context,
      api_data: data.api_data || energyPlan.value.api_data
    }
  }

  function updateCoolingPlan(data: any) {
    coolingPlan.value = { 
      ...coolingPlan.value, 
      ...data
    }
  }

  function updateReviewResult(data: any) {
    reviewResult.value = { 
      ...reviewResult.value, 
      ...data
    }
  }

  function updateFeedback(data: any) {
    feedback.value = { 
      ...feedback.value, 
      ...data
    }
  }

  function updateFinancialAnalysis(data: any) {
    financialAnalysis.value = { 
      ...financialAnalysis.value, 
      ...data
    }
  }

  function updateFinalReport(data: string) {
    finalReport.value = data
  }

  return {
    requirement,
    agentStatus,
    envData,
    energyPlan,
    coolingPlan,
    simulationResult,
    investmentResult,
    reviewResult,
    feedback,
    financialAnalysis,
    finalReport,
    progress,
    updateRequirement,
    updateAgentStatus,
    resetAll,
    updateEnvData,
    updateEnergyPlan,
    updateCoolingPlan,
    updateReviewResult,
    updateFeedback,
    updateFinancialAnalysis,
    updateFinalReport
  }
})
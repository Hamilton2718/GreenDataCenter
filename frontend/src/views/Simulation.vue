<template>
  <div class="simulation-container">
    <el-card class="section-card">
      <template #header>
        <div class="card-header">
          <el-icon><VideoPlay /></el-icon>
          <span>24小时运行仿真</span>
        </div>
      </template>

      <!-- 仿真控制 -->
      <div class="simulation-controls">
        <el-button type="primary" @click="startSimulation" :disabled="isSimulating">
          <el-icon><VideoPlay /></el-icon> 开始仿真
        </el-button>
        <el-button @click="pauseSimulation" :disabled="!isSimulating">
          <el-icon><Pause /></el-icon> 暂停
        </el-button>
        <el-button @click="resetSimulation">
          <el-icon><Refresh /></el-icon> 重置
        </el-button>
        <span class="current-hour">当前小时：{{ currentHour }}/24</span>
      </div>

      <!-- 仿真图表 -->
      <div class="chart-container" ref="chartRef"></div>

      <!-- 验证结果 -->
      <el-card class="result-card" shadow="never" v-if="simulationComplete">
        <template #header>
          <div class="result-header">
            <el-icon><Finished /></el-icon>
            <span>验证结果</span>
          </div>
        </template>

        <el-row :gutter="20">
          <el-col :span="8">
            <el-statistic title="绿电消纳率" :value="simulationResult.greenRatio" suffix="%">
              <template #prefix>
                <el-icon :color="simulationResult.greenRatio >= 90 ? '#67c23a' : '#e6a23c'">
                  <DataLine />
                </el-icon>
              </template>
            </el-statistic>
          </el-col>
          <el-col :span="8">
            <el-statistic title="年均PUE" :value="simulationResult.pue" :precision="2">
              <template #prefix>
                <el-icon :color="simulationResult.pue <= 1.2 ? '#67c23a' : '#e6a23c'">
                  <ColdDrink />
                </el-icon>
              </template>
            </el-statistic>
          </el-col>
          <el-col :span="8">
            <el-statistic title="储能利用率" :value="simulationResult.storageEfficiency" suffix="%">
              <template #prefix>
                <el-icon color="#e6a23c"><Warning /></el-icon>
              </template>
            </el-statistic>
          </el-col>
        </el-row>

        <el-alert
          v-for="(warning, index) in simulationResult.warnings"
          :key="index"
          :title="warning"
          type="warning"
          :closable="false"
          class="warning-item"
        />

        <div class="action-buttons">
          <el-button @click="goBack">返回修改</el-button>
          <el-button type="primary" @click="continueToReport">通过验证，继续</el-button>
        </div>
      </el-card>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useProjectStore } from '../stores/projectStore'
import * as echarts from 'echarts'

const router = useRouter()
const projectStore = useProjectStore()

const chartRef = ref<HTMLElement>()
let chart: echarts.ECharts | null = null

const isSimulating = ref(false)
const currentHour = ref(0)
const simulationComplete = ref(false)

const simulationResult = ref({
  greenRatio: 92,
  pue: 1.18,
  storageEfficiency: 65,
  warnings: ['储能利用率可优化']
})

// 模拟数据
const generateSimulationData = () => {
  const hours = Array.from({ length: 24 }, (_, i) => i)
  const loadData = hours.map(h => 40 + 15 * Math.sin((h - 8) * Math.PI / 12))
  const greenData = hours.map(h => {
    if (h >= 6 && h <= 18) {
      return 35 + 20 * Math.sin((h - 12) * Math.PI / 12)
    }
    return 5
  })
  const storageData = hours.map(h => {
    if (h >= 10 && h <= 14) return -10
    if (h >= 18 && h <= 22) return 8
    return 0
  })
  
  return { hours, loadData, greenData, storageData }
}

const initChart = () => {
  if (!chartRef.value) return
  
  chart = echarts.init(chartRef.value)
  const { hours, loadData, greenData, storageData } = generateSimulationData()
  
  const option = {
    title: {
      text: '24小时运行曲线'
    },
    tooltip: {
      trigger: 'axis'
    },
    legend: {
      data: ['IT负载 (MW)', '绿电供应 (MW)', '储能充放 (MW)']
    },
    xAxis: {
      type: 'category',
      data: hours.map(h => `${h}:00`)
    },
    yAxis: {
      type: 'value'
    },
    series: [
      {
        name: 'IT负载 (MW)',
        type: 'line',
        data: loadData,
        smooth: true,
        lineStyle: { color: '#409eff', width: 3 }
      },
      {
        name: '绿电供应 (MW)',
        type: 'line',
        data: greenData,
        smooth: true,
        lineStyle: { color: '#67c23a', width: 3 }
      },
      {
        name: '储能充放 (MW)',
        type: 'bar',
        data: storageData,
        itemStyle: {
          color: (params: any) => params.value > 0 ? '#e6a23c' : '#f56c6c'
        }
      }
    ]
  }
  
  chart.setOption(option)
}

const startSimulation = () => {
  isSimulating.value = true
  currentHour.value = 0
  
  const interval = setInterval(() => {
    if (currentHour.value >= 23) {
      clearInterval(interval)
      isSimulating.value = false
      simulationComplete.value = true
      projectStore.updateAgentStatus('agent4', true)
      ElMessage.success('仿真完成')
    } else {
      currentHour.value++
    }
  }, 500)
}

const pauseSimulation = () => {
  isSimulating.value = false
  ElMessage.info('仿真已暂停')
}

const resetSimulation = () => {
  isSimulating.value = false
  currentHour.value = 0
  simulationComplete.value = false
  ElMessage.info('仿真已重置')
}

const goBack = () => {
  router.push('/design')
}

const continueToReport = () => {
  projectStore.updateAgentStatus('agent5', true)
  router.push('/report')
}

onMounted(() => {
  initChart()
  window.addEventListener('resize', () => chart?.resize())
})
</script>

<style scoped>
.simulation-container {
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

.simulation-controls {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 24px;
  padding: 16px;
  background-color: #f5f7fa;
  border-radius: 8px;
}

.current-hour {
  margin-left: auto;
  font-weight: bold;
  color: #409eff;
}

.chart-container {
  height: 400px;
  margin: 24px 0;
}

.result-card {
  margin-top: 24px;
}

.result-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: bold;
}

.warning-item {
  margin-top: 16px;
}

.action-buttons {
  display: flex;
  justify-content: flex-end;
  gap: 16px;
  margin-top: 24px;
}
</style>
<template>
  <div class="collaboration-container">
    <!-- Agent协作状态看板 -->
    <el-row :gutter="16" class="agent-dashboard">
      <el-col :span="4.8" v-for="agent in agents" :key="agent.id">
        <el-card :class="['agent-card', { active: agent.status }]" shadow="hover" @click="showAgentDetail(agent.id)">
          <div class="agent-header">
            <el-icon :size="24" :color="agent.status ? '#67c23a' : '#909399'">
              <component :is="agent.icon" />
            </el-icon>
            <el-tag :type="agent.status ? 'success' : 'info'" size="small">
              {{ agent.status ? '已完成' : '等待中' }}
            </el-tag>
          </div>
          <h4>{{ agent.name }}</h4>
          <p class="agent-desc">{{ agent.desc }}</p>
          <el-progress 
            :percentage="agent.progress" 
            :status="agent.status ? 'success' : 'info'"
            :stroke-width="4"
          />
        </el-card>
      </el-col>
    </el-row>

    <!-- 当前激活的Agent详情 -->
    <el-card class="agent-detail" v-if="activeAgent">
      <template #header>
        <div class="detail-header">
          <div class="detail-title">
            <el-icon><component :is="activeAgent.icon" /></el-icon>
            <span>{{ activeAgent.name }} - 输出结果</span>
          </div>
          <el-button type="primary" link @click="activeAgent = null">
            <el-icon><Close /></el-icon>
          </el-button>
        </div>
      </template>

      <!-- Agent 2: 能源规划专家 -->
      <div v-if="activeAgent.id === 2" class="energy-plan">
        <!-- 关键指标卡片 -->
        <el-row :gutter="20">
          <el-col :span="6">
            <el-card shadow="hover">
              <template #header>
                <div class="card-header">
                  <span>光伏容量</span>
                </div>
              </template>
              <div class="card-content">
                <el-progress type="dashboard" :percentage="energyPlan.solar.ratio" :color="'#409EFF'" />
                <div class="card-value">{{ energyPlan.solar.capacity }} MW</div>
                <div class="card-subvalue" v-if="energyPlan.pv_capacity">实际: {{ energyPlan.pv_capacity / 1000 }} MW</div>
              </div>
            </el-card>
          </el-col>
          <el-col :span="6">
            <el-card shadow="hover">
              <template #header>
                <div class="card-header">
                  <span>储能配置</span>
                </div>
              </template>
              <div class="card-content">
                <el-progress type="dashboard" :percentage="energyPlan.storage.ratio" :color="'#67C23A'" />
                <div class="card-value">{{ energyPlan.storage.capacity }} MW</div>
                <div class="card-subvalue">{{ energyPlan.storage.energy }} MWh</div>
                <div class="card-subvalue" v-if="energyPlan.storage_power">实际: {{ energyPlan.storage_power / 1000 }} MW / {{ energyPlan.storage_capacity / 1000 }} MWh</div>
              </div>
            </el-card>
          </el-col>
          <el-col :span="6">
            <el-card shadow="hover">
              <template #header>
                <div class="card-header">
                  <span>绿电长协</span>
                </div>
              </template>
              <div class="card-content">
                <el-progress type="dashboard" :percentage="energyPlan.ppa.ratio" :color="'#E6A23C'" />
                <div class="card-value">{{ energyPlan.ppa.capacity }} MW</div>
                <div class="card-subvalue" v-if="energyPlan.ppa_ratio">实际比例: {{ energyPlan.ppa_ratio }}%</div>
              </div>
            </el-card>
          </el-col>
          <el-col :span="6">
            <el-card shadow="hover">
              <template #header>
                <div class="card-header">
                  <span>电网调峰</span>
                </div>
              </template>
              <div class="card-content">
                <el-progress type="dashboard" :percentage="energyPlan.grid.ratio" :color="'#F56C6C'" />
                <div class="card-value">{{ energyPlan.grid.capacity }} MW</div>
                <div class="card-subvalue" v-if="energyPlan.grid_ratio">实际比例: {{ energyPlan.grid_ratio }}%</div>
              </div>
            </el-card>
          </el-col>
        </el-row>

        <!-- 关键性能指标 -->
        <el-row :gutter="20" style="margin-top: 20px;">
          <el-col :span="6">
            <el-card shadow="hover">
              <template #header>
                <div class="card-header">
                  <span>绿电占比</span>
                </div>
              </template>
              <div class="card-content">
                <el-progress :percentage="energyPlan.estimated_green_ratio" :color="'#409EFF'" />
                <div class="card-value">{{ energyPlan.estimated_green_ratio }}%</div>
                <div class="card-subvalue">目标: 90%</div>
              </div>
            </el-card>
          </el-col>
          <el-col :span="6">
            <el-card shadow="hover">
              <template #header>
                <div class="card-header">
                  <span>自发自用率</span>
                </div>
              </template>
              <div class="card-content">
                <el-progress :percentage="energyPlan.estimated_self_consumption" :color="'#67C23A'" />
                <div class="card-value">{{ energyPlan.estimated_self_consumption }}%</div>
              </div>
            </el-card>
          </el-col>
          <el-col :span="6">
            <el-card shadow="hover">
              <template #header>
                <div class="card-header">
                  <span>风电容量</span>
                </div>
              </template>
              <div class="card-content">
                <el-progress type="dashboard" :percentage="0" :color="'#909399'" />
                <div class="card-value" v-if="energyPlan.wind_capacity">{{ energyPlan.wind_capacity / 1000 }} MW</div>
                <div class="card-value" v-else>0 MW</div>
              </div>
            </el-card>
          </el-col>
          <el-col :span="6">
            <el-card shadow="hover">
              <template #header>
                <div class="card-header">
                  <span>收益预估</span>
                </div>
              </template>
              <div class="card-content">
                <el-progress type="dashboard" :percentage="75" :color="'#E6A23C'" />
                <div class="card-value">¥175万/年</div>
                <div class="card-subvalue">储能套利 + 辅助服务</div>
              </div>
            </el-card>
          </el-col>
        </el-row>

        <div class="energy-chart">
          <v-chart class="chart" :option="energyChartOption" />
        </div>

        <el-divider />

        <!-- 配置摘要 -->
        <div class="config-summary" v-if="energyPlan.pv_capacity || energyPlan.storage_capacity">
          <h4>📊 配置摘要</h4>
          <el-descriptions :column="2" border>
            <el-descriptions-item label="光伏装机容量">
              {{ energyPlan.pv_capacity }} kW
            </el-descriptions-item>
            <el-descriptions-item label="风电装机容量">
              {{ energyPlan.wind_capacity }} kW
            </el-descriptions-item>
            <el-descriptions-item label="储能容量">
              {{ energyPlan.storage_capacity }} kWh
            </el-descriptions-item>
            <el-descriptions-item label="储能功率">
              {{ energyPlan.storage_power }} kW
            </el-descriptions-item>
            <el-descriptions-item label="绿电长协比例">
              {{ energyPlan.ppa_ratio }}%
            </el-descriptions-item>
            <el-descriptions-item label="电网调峰比例">
              {{ energyPlan.grid_ratio }}%
            </el-descriptions-item>
            <el-descriptions-item label="预计自发自用率">
              {{ energyPlan.estimated_self_consumption }}%
            </el-descriptions-item>
            <el-descriptions-item label="预计绿电占比">
              {{ energyPlan.estimated_green_ratio }}%
            </el-descriptions-item>
          </el-descriptions>
        </div>

        <el-divider />

        <!-- Markdown报告 -->
        <div class="markdown-report" v-if="energyPlan.llm_report">
          <div class="report-header">
            <h4>📋 能源规划报告</h4>
            <el-button-group>
              <el-button size="small" :type="reportViewMode === 'preview' ? 'primary' : 'default'" @click="reportViewMode = 'preview'">预览</el-button>
              <el-button size="small" :type="reportViewMode === 'code' ? 'primary' : 'default'" @click="reportViewMode = 'code'">源码</el-button>
              <el-button size="small" type="info" @click="exportReport('md')">导出MD</el-button>
              <el-button size="small" type="info" @click="exportReport('html')">导出HTML</el-button>
              <el-button size="small" type="primary" @click="regenerateReport">重新生成</el-button>
            </el-button-group>
          </div>
          <div class="markdown-content" v-if="reportViewMode === 'preview'" v-html="renderMarkdown(energyPlan.llm_report)"></div>
          <el-input v-else type="textarea" :value="energyPlan.llm_report" :rows="15" readonly style="font-family: monospace;"></el-input>
        </div>
        <div class="markdown-report" v-else>
          <h4>📋 能源规划报告</h4>
          <div class="markdown-content">
            <p>报告生成中，请稍候...</p>
            <p>或点击下方按钮重新生成报告</p>
            <el-button type="primary" @click="regenerateReport" style="margin-top: 10px;">重新生成报告</el-button>
          </div>
        </div>

        <!-- 目标对比 -->
        <div class="target-comparison" style="margin-top: 20px;">
          <h4>🎯 目标对比</h4>
          <el-row :gutter="20">
            <el-col :span="12">
              <el-card shadow="hover">
                <template #header>
                  <span>绿电占比</span>
                </template>
                <el-progress :percentage="energyPlan.estimated_green_ratio" :color="'#409EFF'" />
                <div class="target-info">
                  <span>当前: {{ energyPlan.estimated_green_ratio }}%</span>
                  <span>目标: 90%</span>
                  <span :class="energyPlan.estimated_green_ratio >= 90 ? 'success' : 'warning'">
                    {{ energyPlan.estimated_green_ratio >= 90 ? '✅ 达标' : '⚠️ 未达标' }}
                  </span>
                </div>
              </el-card>
            </el-col>
            <el-col :span="12">
              <el-card shadow="hover">
                <template #header>
                  <span>自发自用率</span>
                </template>
                <el-progress :percentage="energyPlan.estimated_self_consumption" :color="'#67C23A'" />
                <div class="target-info">
                  <span>当前: {{ energyPlan.estimated_self_consumption }}%</span>
                  <span>目标: 80%</span>
                  <span :class="energyPlan.estimated_self_consumption >= 80 ? 'success' : 'warning'">
                    {{ energyPlan.estimated_self_consumption >= 80 ? '✅ 达标' : '⚠️ 未达标' }}
                  </span>
                </div>
              </el-card>
            </el-col>
          </el-row>
        </div>

        <!-- 优化建议 -->
        <div class="optimization-suggestions" style="margin-top: 20px;">
          <h4>💡 优化建议</h4>
          <el-card shadow="hover">
            <el-list>
              <el-list-item v-for="(suggestion, index) in optimizationSuggestions" :key="index">
                <template #prefix>
                  <el-tag size="small" :type="suggestion.type">{{ suggestion.type }}</el-tag>
                </template>
                {{ suggestion.content }}
              </el-list-item>
            </el-list>
          </el-card>
        </div>
      </div>

      <!-- Agent 3: 制冷架构专家 -->
      <div v-if="activeAgent.id === 3" class="cooling-plan">
        <el-row :gutter="20">
          <el-col :span="12">
            <el-card shadow="never">
              <template #header>
                <span>制冷技术路线</span>
              </template>
              <el-steps direction="vertical" :active="2">
                <el-step title="自然冷却" description="利用当地年均8.2℃低温，全年约200天可实现自然冷却" />
                <el-step title="间接蒸发冷却" description="过渡季节开启，降低压缩机运行时间" />
                <el-step title="液冷机柜" description="针对30kW高密度机柜，采用冷板式液冷" />
              </el-steps>
            </el-card>
          </el-col>
          <el-col :span="12">
            <el-card shadow="never">
              <template #header>
                <span>预计能效</span>
              </template>
              <div class="pue-display">
                <h2>1.18</h2>
                <p>年均PUE</p>
                <el-progress type="dashboard" :percentage="85" :width="150" color="#409eff">
                  <template #default>
                    <span class="pue-target">目标 1.2</span>
                  </template>
                </el-progress>
              </div>
            </el-card>
          </el-col>
        </el-row>
      </div>

      <div class="agent-actions" v-if="activeAgent">
        <el-button @click="activeAgent = null">关闭</el-button>
        <el-button type="primary" @click="approveAgent">确认方案</el-button>
        <el-button type="warning" @click="rejectAgent">提出修改意见</el-button>
      </div>
    </el-card>

    <!-- 操作按钮 -->
    <div class="action-buttons">
      <el-button @click="router.push('/requirement')">
        <el-icon><ArrowLeft /></el-icon> 上一步
      </el-button>
      <el-button type="primary" @click="submitToSimulation">
        <el-icon><VideoPlay /></el-icon> 提交仿真验证
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useProjectStore } from '../stores/projectStore'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { PieChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
} from 'echarts/components'

use([
  CanvasRenderer,
  PieChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
])

const router = useRouter()
const projectStore = useProjectStore()
const activeAgent = ref<any>(null)

// Agent列表
const agents = ref([
  {
    id: 1,
    name: '需求解析专家',
    desc: '解析用户需求，量化约束条件',
    icon: 'Document',
    status: projectStore.agentStatus.agent1,
    progress: 100
  },
  {
    id: 2,
    name: '能源规划专家',
    desc: '计算绿电配比和储能容量',
    icon: 'DataLine',
    status: projectStore.agentStatus.agent2,
    progress: projectStore.agentStatus.agent2 ? 100 : 60
  },
  {
    id: 3,
    name: '制冷架构专家',
    desc: '设计制冷方案，计算PUE',
    icon: 'ColdDrink',
    status: projectStore.agentStatus.agent3,
    progress: projectStore.agentStatus.agent3 ? 100 : 30
  },
  {
    id: 4,
    name: '仿真验证专家',
    desc: '模拟24小时运行情况',
    icon: 'VideoPlay',
    status: projectStore.agentStatus.agent4,
    progress: projectStore.agentStatus.agent4 ? 100 : 0
  },
  {
    id: 5,
    name: '投资决策专家',
    desc: '评估成本和投资回报',
    icon: 'Money',
    status: projectStore.agentStatus.agent5,
    progress: projectStore.agentStatus.agent5 ? 100 : 0
  }
])

const energyPlan = computed(() => projectStore.energyPlan)

// 报告显示模式
const reportViewMode = ref('preview')

// 优化建议
const optimizationSuggestions = ref([
  { type: 'primary', content: '增加光伏装机容量，提高自发自用率' },
  { type: 'success', content: '优化储能充放电策略，提高套利收益' },
  { type: 'warning', content: '考虑参与虚拟电厂，增加辅助服务收益' },
  { type: 'info', content: '探索绿证交易，进一步提高绿电占比' }
])

// 能源配比图表配置
const energyChartOption = computed(() => ({
  title: {
    text: '能源配比方案',
    left: 'center'
  },
  tooltip: {
    trigger: 'item',
    formatter: '{a} <br/>{b}: {c}%'
  },
  series: [
    {
      name: '能源配比',
      type: 'pie',
      radius: '50%',
      data: [
        { value: energyPlan.value.solar.ratio, name: '分布式光伏' },
        { value: energyPlan.value.storage.ratio, name: '储能' },
        { value: energyPlan.value.ppa.ratio, name: '绿电长协' },
        { value: energyPlan.value.grid.ratio, name: '电网调峰' }
      ],
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowOffsetX: 0,
          shadowColor: 'rgba(0, 0, 0, 0.5)'
        }
      }
    }
  ]
}))

// 导出报告
const exportReport = (format) => {
  if (!energyPlan.value.llm_report) return
  
  let content = energyPlan.value.llm_report
  let filename = '能源规划报告'
  let mimeType = 'text/plain'
  
  if (format === 'html') {
    content = renderMarkdown(energyPlan.value.llm_report)
    content = `<html><head><style>body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; } h1 { color: #333; } h2 { color: #555; } ul { margin: 10px 0; } li { margin: 5px 0; } </style></head><body>${content}</body></html>`
    filename += '.html'
    mimeType = 'text/html'
  } else {
    filename += '.md'
  }
  
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
  
  ElMessage.success(`报告已导出为 ${filename}`)
}

// 获取能源规划数据
const fetchEnergyPlan = async () => {
  try {
    ElMessage.info('正在获取能源规划数据...')
    const response = await fetch('/api/agent2/energy-plan', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(projectStore.requirement)
    })
    const data = await response.json()
    if (data.success) {
      projectStore.updateEnergyPlan(data.data)
      ElMessage.success('能源规划数据获取成功！')
    } else {
      ElMessage.error('获取能源规划数据失败：' + (data.error || '未知错误'))
    }
  } catch (error) {
    console.error('获取能源规划数据失败：', error)
    ElMessage.error('获取能源规划数据失败，请检查后端服务是否运行')
  }
}

// 重新生成报告
const regenerateReport = () => {
  ElMessage.info('正在重新生成能源规划报告...')
  fetchEnergyPlan()
}

// 显示Agent详情
const showAgentDetail = (id: number) => {
  activeAgent.value = agents.value.find(a => a.id === id)
}

// 渲染Markdown为HTML
const renderMarkdown = (markdown: string): string => {
  // 简单的Markdown解析
  let html = markdown
    // 标题
    .replace(/^# (.*$)/gm, '<h1>$1</h1>')
    .replace(/^## (.*$)/gm, '<h2>$1</h2>')
    .replace(/^### (.*$)/gm, '<h3>$1</h3>')
    // 列表
    .replace(/^- (.*$)/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>')
    // 段落
    .replace(/^(?!<h|.*<ul|.*<li).*$/gm, '<p>$&</p>')
  return html
}

// 确认Agent方案
const approveAgent = () => {
  if (activeAgent.value) {
    projectStore.updateAgentStatus(`agent${activeAgent.value.id}` as any, true)
    activeAgent.value.status = true
    activeAgent.value.progress = 100
    ElMessage.success(`${activeAgent.value.name} 方案已确认`)
    activeAgent.value = null
  }
}

// 提出修改意见
const rejectAgent = () => {
  ElMessage.warning('修改意见已发送给Agent')
}

// 提交仿真验证
const submitToSimulation = () => {
  // 模拟Agent 4开始工作
  projectStore.updateAgentStatus('agent4', true)
  router.push('/simulation')
}

onMounted(() => {
  // 模拟Agent 2和3正在工作
  setTimeout(() => {
    projectStore.updateAgentStatus('agent2', true)
    agents.value[1].status = true
    agents.value[1].progress = 100
    // 获取能源规划数据
    fetchEnergyPlan()
  }, 2000)
})
</script>

<style scoped>
.collaboration-container {
  max-width: 1400px;
  margin: 0 auto;
}

.agent-dashboard {
  margin-bottom: 24px;
}

.agent-card {
  cursor: pointer;
  transition: all 0.3s;
}

.agent-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

/* 卡片样式 */
.card-header {
  font-weight: bold;
  font-size: 14px;
}

.card-content {
  text-align: center;
  padding: 20px 0;
}

.card-value {
  font-size: 18px;
  font-weight: bold;
  margin-top: 10px;
}

.card-subvalue {
  font-size: 14px;
  color: #606266;
  margin-top: 5px;
}

/* Markdown报告样式 */
.markdown-report {
  margin-top: 20px;
}

.markdown-content {
  background: #f5f7fa;
  padding: 20px;
  border-radius: 8px;
  line-height: 1.6;
}

.markdown-content h1 {
  font-size: 20px;
  font-weight: bold;
  margin-bottom: 15px;
  color: #303133;
}

.markdown-content h2 {
  font-size: 16px;
  font-weight: bold;
  margin: 15px 0 10px;
  color: #409EFF;
}

.markdown-content h3 {
  font-size: 14px;
  font-weight: bold;
  margin: 10px 0;
  color: #67C23A;
}

.markdown-content p {
  margin: 10px 0;
  color: #606266;
}

.markdown-content ul {
  margin: 10px 0;
  padding-left: 20px;
}

.markdown-content li {
  margin: 5px 0;
  color: #606266;
}

/* 报告头部样式 */
.report-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

/* 配置摘要样式 */
.config-summary {
  margin: 20px 0;
}

/* 目标对比样式 */
.target-info {
  display: flex;
  justify-content: space-between;
  margin-top: 10px;
  font-size: 14px;
}

.target-info .success {
  color: #67C23A;
}

.target-info .warning {
  color: #E6A23C;
}

/* 优化建议样式 */
.optimization-suggestions {
  margin: 20px 0;
}

.agent-card.active {
  border: 2px solid #67c23a;
}

.agent-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.agent-desc {
  color: #909399;
  font-size: 0.9rem;
  margin: 8px 0;
  height: 40px;
}

.agent-detail {
  margin-bottom: 24px;
}

.detail-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.detail-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 1.2rem;
  font-weight: bold;
}

.energy-plan {
  padding: 16px;
}

.energy-chart {
  height: 300px;
  margin-top: 20px;
}

.cooling-plan {
  padding: 16px;
}

.pue-display {
  text-align: center;
}

.agent-actions {
  margin-top: 24px;
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

.action-buttons {
  margin-top: 24px;
  display: flex;
  justify-content: space-between;
}
</style>
<template>
  <div class="report-container">
    <el-card class="section-card">
      <template #header>
        <div class="card-header">
          <el-icon><Document /></el-icon>
          <span>数据中心绿电消纳规划设计建议书</span>
        </div>
      </template>

      <!-- 方案摘要 -->
      <el-row :gutter="20" class="summary-section">
        <el-col :span="6">
          <el-statistic title="总容量" :value="40" suffix="MW" />
        </el-col>
        <el-col :span="6">
          <el-statistic title="绿电比例" :value="85" suffix="%" />
        </el-col>
        <el-col :span="6">
          <el-statistic title="年均PUE" :value="1.18" :precision="2" />
        </el-col>
        <el-col :span="6">
          <el-statistic title="总投资" :value="28000" suffix="万元" />
        </el-col>
      </el-row>

      <!-- 帕累托曲线 -->
      <el-row :gutter="20" class="pareto-section">
        <el-col :span="24">
          <h3>可靠性 vs 经济性曲线</h3>
          <div class="chart-container" ref="paretoChartRef"></div>
          <p class="chart-desc">当前方案在帕累托最优前沿面上，达到最佳平衡点</p>
        </el-col>
      </el-row>

      <!-- 详细配置清单 -->
      <el-divider />
      
      <el-row :gutter="20">
        <el-col :span="24">
          <h3>详细配置清单</h3>
          <el-descriptions :column="2" border>
            <el-descriptions-item label="制冷方案" label-width="150px">
              间接蒸发冷却 + 液冷机柜
            </el-descriptions-item>
            <el-descriptions-item label="预计PUE">
              1.18
            </el-descriptions-item>
            <el-descriptions-item label="分布式光伏">
              3.6 MW 屋顶光伏 + 8 MW 车棚光伏
            </el-descriptions-item>
            <el-descriptions-item label="年发电量">
              约1500万 kWh
            </el-descriptions-item>
            <el-descriptions-item label="储能系统">
              12 MWh 磷酸铁锂储能系统
            </el-descriptions-item>
            <el-descriptions-item label="储能配置">
              8 MW / 4小时
            </el-descriptions-item>
            <el-descriptions-item label="绿电长协">
              10年期限，120 GWh/年
            </el-descriptions-item>
            <el-descriptions-item label="年省碳排">
              12,000 吨 CO₂
            </el-descriptions-item>
          </el-descriptions>
        </el-col>
      </el-row>

      <!-- 投资分析 -->
      <el-divider />

      <el-row :gutter="20">
        <el-col :span="12">
          <el-card shadow="never">
            <template #header>
              <span>CAPEX（建设成本）</span>
            </template>
            <el-progress type="circle" :percentage="45" :width="120" color="#409eff">
              <span>45%</span>
            </el-progress>
            <p class="cost-detail">制冷系统: 6,500万</p>
            <p class="cost-detail">光伏系统: 4,200万</p>
            <p class="cost-detail">储能系统: 8,500万</p>
            <p class="cost-detail">配电系统: 8,800万</p>
          </el-card>
        </el-col>
        <el-col :span="12">
          <el-card shadow="never">
            <template #header>
              <span>OPEX（运行成本）</span>
            </template>
            <el-progress type="circle" :percentage="35" :width="120" color="#67c23a">
              <span>35%</span>
            </el-progress>
            <p class="cost-detail">电费支出: 2,800万/年</p>
            <p class="cost-detail">运维费用: 1,200万/年</p>
            <p class="cost-detail">绿电采购: 3,500万/年</p>
            <p class="cost-detail">碳交易收入: -800万/年</p>
          </el-card>
        </el-col>
      </el-row>

      <!-- 最终效果 -->
      <el-divider />

      <el-row :gutter="20" class="final-effect">
        <el-col :span="24">
          <h3>在当前建议下达到的最终效果</h3>
          <el-table :data="effectData" border stripe>
            <el-table-column prop="indicator" label="指标" width="200" />
            <el-table-column prop="target" label="目标值" />
            <el-table-column prop="actual" label="实际达到">
              <template #default="{ row }">
                <span :class="row.actual >= row.target ? 'success' : 'warning'">
                  {{ row.actual }}
                  <el-icon v-if="row.actual >= row.target"><SuccessFilled /></el-icon>
                  <el-icon v-else><WarningFilled /></el-icon>
                </span>
              </template>
            </el-table-column>
          </el-table>
        </el-col>
      </el-row>

      <!-- 操作按钮 -->
      <div class="action-buttons">
        <el-button type="success" @click="downloadReport">
          <el-icon><Download /></el-icon> 下载PDF报告
        </el-button>
        <el-button type="primary" @click="shareReport">
          <el-icon><Share /></el-icon> 分享
        </el-button>
        <el-button @click="restart">
          <el-icon><Refresh /></el-icon> 重新规划
        </el-button>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'

const router = useRouter()
const paretoChartRef = ref<HTMLElement>()

const effectData = ref([
  { indicator: '绿电消纳率', target: '90%', actual: '92%' },
  { indicator: '年均PUE', target: '1.20', actual: '1.18' },
  { indicator: '投资回报期', target: '7年', actual: '6.5年' },
  { indicator: '年碳减排', target: '10,000吨', actual: '12,000吨' }
])

const initParetoChart = () => {
  if (!paretoChartRef.value) return
  
  const chart = echarts.init(paretoChartRef.value)
  
  // 生成帕累托前沿数据
  const reliability = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
  const cost = [18000, 21000, 24000, 28000, 35000, 45000, 60000]
  
  const option = {
    tooltip: {
      trigger: 'item',
      formatter: '可靠性: {c0}%<br/>成本: {c1}万元'
    },
    xAxis: {
      type: 'category',
      name: '可靠性 (%)',
      data: reliability.map(r => r * 100)
    },
    yAxis: {
      type: 'value',
      name: '成本 (万元)'
    },
    series: [
      {
        data: cost,
        type: 'line',
        smooth: true,
        lineStyle: {
          color: '#409eff',
          width: 3
        },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(64,158,255,0.3)' },
            { offset: 1, color: 'rgba(64,158,255,0.1)' }
          ])
        },
        markPoint: {
          data: [
            { type: 'max', name: '最大值' },
            { type: 'min', name: '最小值' }
          ]
        },
        markLine: {
          data: [
            { type: 'average', name: '平均值' }
          ]
        }
      },
      {
        data: reliability.map((r, i) => ({
          value: [r * 100, cost[i]],
          symbol: 'circle',
          symbolSize: 10,
          itemStyle: { color: '#67c23a' }
        })),
        type: 'scatter',
        name: '可行方案',
        tooltip: { show: false }
      }
    ]
  }
  
  chart.setOption(option)
  window.addEventListener('resize', () => chart.resize())
}

const downloadReport = () => {
  ElMessage.success('报告下载中...')
  setTimeout(() => {
    ElMessage.success('报告下载完成')
  }, 1500)
}

const shareReport = () => {
  ElMessage.success('分享链接已复制到剪贴板')
}

const restart = () => {
  ElMessage.info('重新开始规划')
  router.push('/requirement')
}

onMounted(() => {
  initParetoChart()
})
</script>

<style scoped>
.report-container {
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
  font-size: 1.2rem;
}

.summary-section {
  margin-bottom: 32px;
}

.pareto-section {
  margin: 32px 0;
}

.chart-container {
  height: 300px;
  margin: 16px 0;
}

.chart-desc {
  color: #909399;
  font-size: 0.9rem;
  text-align: center;
  margin-top: 8px;
}

.cost-detail {
  margin: 8px 0;
  color: #606266;
}

.final-effect {
  margin: 24px 0;
}

.success {
  color: #67c23a;
}

.warning {
  color: #e6a23c;
}

.action-buttons {
  display: flex;
  justify-content: center;
  gap: 16px;
  margin-top: 32px;
}
</style>
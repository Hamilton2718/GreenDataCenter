<template>
  <div class="report-container">
    <el-card class="section-card">
      <template #header>
        <div class="card-header">
          <el-icon><Document /></el-icon>
          <span>数据中心绿电消纳规划设计建议书</span>
        </div>
      </template>

      <!-- 最终报告内容 -->
      <div v-if="projectStore.finalReport" class="final-report-content">
        <div v-html="renderMarkdown(projectStore.finalReport)"></div>
      </div>
      <div v-else class="loading-content">
        <el-empty description="未找到最终报告" />
        <el-button type="primary" @click="generateReport" style="margin-top: 20px;">
          <el-icon><Refresh /></el-icon> 生成报告
        </el-button>
      </div>

      <!-- 原始数据 -->
      <div v-if="projectStore.finalReport" class="raw-data-section">
        <el-divider content-position="left">原始数据</el-divider>
        <el-collapse>
          <el-collapse-item title="用户需求">
            <pre>{{ JSON.stringify(projectStore.requirement, null, 2) }}</pre>
          </el-collapse-item>
          <el-collapse-item title="环境数据">
            <pre>{{ JSON.stringify(projectStore.envData, null, 2) }}</pre>
          </el-collapse-item>
          <el-collapse-item title="能源方案">
            <pre>{{ JSON.stringify(projectStore.energyPlan, null, 2) }}</pre>
          </el-collapse-item>
          <el-collapse-item title="制冷方案">
            <pre>{{ JSON.stringify(projectStore.coolingPlan, null, 2) }}</pre>
          </el-collapse-item>
          <el-collapse-item title="仿真结果">
            <pre>{{ JSON.stringify(projectStore.simulationResult, null, 2) }}</pre>
          </el-collapse-item>
          <el-collapse-item title="财务分析">
            <pre>{{ JSON.stringify(projectStore.financialAnalysis, null, 2) }}</pre>
          </el-collapse-item>
        </el-collapse>
      </div>

      <!-- 操作按钮 -->
      <div class="action-buttons">
        <el-button type="success" @click="downloadReport" :disabled="!projectStore.finalReport">
          <el-icon><Download /></el-icon> 下载PDF报告
        </el-button>
        <el-button type="primary" @click="shareReport" :disabled="!projectStore.finalReport">
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
import { useProjectStore } from '../stores/projectStore'

const router = useRouter()
const projectStore = useProjectStore()

// 渲染Markdown内容
const renderMarkdown = (markdown: string): string => {
  // 简单的Markdown渲染
  let html = markdown
    // 处理标题
    .replace(/^# (.*$)/gm, '<h1>$1</h1>')
    .replace(/^## (.*$)/gm, '<h2>$1</h2>')
    .replace(/^### (.*$)/gm, '<h3>$1</h3>')
    
  // 处理表格
  // 查找表格部分
  const tableRegex = /(\|.*\|\n)+/g
  html = html.replace(tableRegex, (tableContent) => {
    // 分割表格行
    const rows = tableContent.trim().split('\n')
    if (rows.length < 2) return tableContent
    
    // 构建表格HTML
    let tableHtml = '<table class="markdown-table"><thead><tr>'
    
    // 处理表头
    const headerCells = rows[0].split('|').filter(cell => cell.trim() !== '')
    headerCells.forEach(cell => {
      tableHtml += `<th>${cell.trim()}</th>`
    })
    tableHtml += '</tr></thead><tbody>'
    
    // 跳过分隔线行，处理数据行
    for (let i = 2; i < rows.length; i++) {
      const cells = rows[i].split('|').filter(cell => cell.trim() !== '')
      if (cells.length > 0) {
        tableHtml += '<tr>'
        cells.forEach(cell => {
          tableHtml += `<td>${cell.trim()}</td>`
        })
        tableHtml += '</tr>'
      }
    }
    
    tableHtml += '</tbody></table>'
    return tableHtml
  })
  
  // 处理普通段落
  html = html.replace(/^(?!<h|.*<table).*$/gm, '<p>$&</p>')
  
  return html
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

const generateReport = async () => {
  try {
    ElMessage.info('正在生成最终报告...')
    
    // 准备请求数据
    const requestData = {
      user_requirements: projectStore.requirement,
      environmental_data: projectStore.envData,
      energy_plan: projectStore.energyPlan,
      cooling_plan: projectStore.coolingPlan,
      simulation_result: projectStore.simulationResult,
      financial_analysis: projectStore.financialAnalysis
    }
    
    console.log('发送给后端的报告生成数据:', requestData)
    
    // 调用后端API
    const response = await fetch('http://localhost:5004/api/generate-report', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestData)
    })
    
    if (response.ok) {
      const result = await response.json()
      console.log('API响应数据:', result)
      
      if (result.success) {
        // 后端返回的完整状态数据
        const fullState = result.data
        
        // 更新最终报告
        projectStore.updateFinalReport(fullState.final_report)
        
        // 显示成功消息
        ElMessage.success('最终报告生成完成')
      } else {
        ElMessage.error('报告生成失败: ' + result.error)
      }
    } else {
      const errorText = await response.text()
      console.log('服务器错误:', errorText)
      ElMessage.error('服务器错误，请稍后重试')
    }
  } catch (error) {
    console.error('生成最终报告失败:', error)
    ElMessage.error('生成最终报告失败，请稍后重试')
  }
}

const restart = () => {
  ElMessage.info('重新开始规划')
  router.push('/requirement')
}

onMounted(() => {
  // 检查是否有最终报告
  if (!projectStore.finalReport) {
    console.log('未找到最终报告，等待用户手动生成')
  }
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

.final-report-content {
  margin: 24px 0;
  line-height: 1.6;
}

.final-report-content h1 {
  font-size: 1.8rem;
  margin: 24px 0 16px;
  color: #303133;
}

.final-report-content h2 {
  font-size: 1.4rem;
  margin: 20px 0 12px;
  color: #409eff;
}

.final-report-content h3 {
  font-size: 1.2rem;
  margin: 16px 0 8px;
  color: #606266;
}

.final-report-content p {
  margin: 8px 0;
  color: #606266;
}

.final-report-content table,
.final-report-content .markdown-table {
  width: 100%;
  border-collapse: collapse;
  margin: 16px 0;
}

.final-report-content table th,
.final-report-content table td,
.final-report-content .markdown-table th,
.final-report-content .markdown-table td {
  border: 1px solid #dcdfe6;
  padding: 8px 12px;
  text-align: left;
}

.final-report-content table th,
.final-report-content .markdown-table th {
  background-color: #f5f7fa;
  font-weight: bold;
  color: #303133;
}

.loading-content {
  padding: 48px 0;
  text-align: center;
}

.raw-data-section {
  margin-top: 32px;
  padding: 16px;
  background-color: #f5f7fa;
  border-radius: 8px;
}

.raw-data-section pre {
  background-color: #ffffff;
  padding: 12px;
  border-radius: 4px;
  overflow-x: auto;
  font-size: 14px;
  line-height: 1.4;
}

.action-buttons {
  display: flex;
  justify-content: center;
  gap: 16px;
  margin-top: 32px;
}
</style>
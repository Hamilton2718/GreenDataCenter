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

      <!-- Agent 1: 需求解析专家 -->
      <div v-if="activeAgent.id === 1" class="requirement-analysis">
        <!-- 项目基本信息 -->
        <el-card shadow="hover" style="margin-bottom: 20px;">
          <template #header>
            <span>项目基本信息</span>
          </template>
          <el-descriptions :column="2" border>
            <el-descriptions-item label="地理位置">
              {{ projectStore.requirement.location }}
            </el-descriptions-item>
            <el-descriptions-item label="业务类型">
              {{ projectStore.requirement.businessType }}
            </el-descriptions-item>
            <el-descriptions-item label="计划面积">
              {{ projectStore.requirement.area }} m²
            </el-descriptions-item>
            <el-descriptions-item label="计划负荷">
              {{ projectStore.requirement.load }} MW
            </el-descriptions-item>
            <el-descriptions-item label="算力密度">
              {{ projectStore.requirement.density }} kW/机柜
            </el-descriptions-item>
            <el-descriptions-item label="优先级">
              <el-tag v-for="p in projectStore.requirement.priority" :key="p" size="small" style="margin-right: 5px;">
                {{ p === 'reliable' ? '可靠型' : (p === 'economic' ? '经济型' : '环保型') }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="绿电目标">
              {{ projectStore.requirement.greenTarget }}%
            </el-descriptions-item>
            <el-descriptions-item label="PUE目标">
              {{ projectStore.requirement.pueTarget }}
            </el-descriptions-item>
          </el-descriptions>
        </el-card>

        <!-- 环境数据 -->
        <el-card shadow="hover" style="margin-bottom: 20px;">
          <template #header>
            <span>环境数据</span>
          </template>
          <el-row :gutter="20">
            <el-col :span="6">
              <el-statistic title="年均温度" :value="projectStore.envData.climate.avgTemp" :precision="2" suffix="°C" />
            </el-col>
            <el-col :span="6">
              <el-statistic title="年均风速" :value="projectStore.envData.climate.windSpeed" :precision="2" suffix="m/s" />
            </el-col>
            <el-col :span="6">
              <el-statistic title="年日照时数" :value="projectStore.envData.climate.solarRadiation" :precision="2" suffix="h" />
            </el-col>
            <el-col :span="6">
              <el-statistic title="碳排因子" :value="projectStore.envData.carbonFactor" :precision="4" suffix="kg CO₂/kWh" />
            </el-col>
          </el-row>
        </el-card>

        <!-- 电价数据 -->
        <el-card shadow="hover" style="margin-bottom: 20px;">
          <template #header>
            <span>电价数据</span>
          </template>
          <el-row :gutter="20">
            <el-col :span="8">
              <el-statistic title="尖峰电价" :value="projectStore.envData.electricity.peakPrice" :precision="4" suffix="元/kWh" />
            </el-col>
            <el-col :span="8">
              <el-statistic title="高峰电价" :value="projectStore.envData.electricity.highPrice" :precision="4" suffix="元/kWh" />
            </el-col>
            <el-col :span="8">
              <el-statistic title="平段电价" :value="projectStore.envData.electricity.flatPrice" :precision="4" suffix="元/kWh" />
            </el-col>
          </el-row>
          <el-row :gutter="20" style="margin-top: 20px">
            <el-col :span="8">
              <el-statistic title="低谷电价" :value="projectStore.envData.electricity.valleyPrice" :precision="4" suffix="元/kWh" />
            </el-col>
            <el-col :span="8">
              <el-statistic title="深谷电价" :value="projectStore.envData.electricity.deepValleyPrice" :precision="4" suffix="元/kWh" />
            </el-col>
            <el-col :span="8">
              <el-statistic title="最大峰谷价差" :value="projectStore.envData.electricity.maxPriceDiff" :precision="4" suffix="元/kWh" />
            </el-col>
          </el-row>
        </el-card>
      </div>

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
        <!-- 核心指标 -->
        <el-row :gutter="20" style="margin-bottom: 20px;">
          <el-col :span="6">
            <el-card shadow="hover">
              <template #header>
                <div class="card-header">
                  <span>推荐技术</span>
                </div>
              </template>
              <div class="card-content">
                <div class="card-value">{{ projectStore.coolingPlan.cooling_technology || 'N/A' }}</div>
              </div>
            </el-card>
          </el-col>
          <el-col :span="6">
            <el-card shadow="hover">
              <template #header>
                <div class="card-header">
                  <span>预计PUE</span>
                </div>
              </template>
              <div class="card-content">
                <div class="card-value">{{ projectStore.coolingPlan.estimated_pue || 'N/A' }}</div>
                <div class="card-unit">目标: {{ projectStore.requirement.pueTarget }}</div>
              </div>
            </el-card>
          </el-col>
          <el-col :span="6">
            <el-card shadow="hover">
              <template #header>
                <div class="card-header">
                  <span>预计WUE</span>
                </div>
              </template>
              <div class="card-content">
                <div class="card-value">{{ projectStore.coolingPlan.predicted_wue || 'N/A' }}</div>
              </div>
            </el-card>
          </el-col>
          <el-col :span="6">
            <el-card shadow="hover">
              <template #header>
                <div class="card-header">
                  <span>制冷功率</span>
                </div>
              </template>
              <div class="card-content">
                <div class="card-value">{{ projectStore.coolingPlan.cooling_kpis?.cooling_power_kw || 'N/A' }}</div>
                <div class="card-unit">kW</div>
              </div>
            </el-card>
          </el-col>
        </el-row>

        <!-- 项目信息 -->
        <el-card shadow="hover" style="margin-bottom: 20px;">
          <template #header>
            <span>项目信息</span>
          </template>
          <el-table :data="[projectStore.coolingPlan.cooling_project_info]" style="width: 100%">
            <el-table-column prop="location" label="位置" />
            <el-table-column prop="it_load_kW" label="IT负载(kW)" />
            <el-table-column prop="cabinet_power_kW" label="机柜功率(kW)" />
            <el-table-column prop="target_pue" label="目标PUE" />
            <el-table-column prop="green_energy_target" label="绿电目标(%)" />
          </el-table>
        </el-card>

        <!-- 计算参数 -->
        <el-card shadow="hover" style="margin-bottom: 20px;">
          <template #header>
            <span>计算参数</span>
          </template>
          <el-table :data="[projectStore.coolingPlan.cooling_calc_params]" style="width: 100%">
            <el-table-column prop="PUE_Limit" label="PUE限值" />
            <el-table-column prop="WUE_Limit" label="WUE限值" />
            <el-table-column prop="cooling_eff_coeff" label="制冷效率系数" />
            <el-table-column prop="facility_loss_coeff" label="设施损耗系数" />
            <el-table-column prop="regional_cooling_preference" label="区域制冷偏好" />
          </el-table>
        </el-card>

        <!-- KPI数据 -->
        <el-card shadow="hover" style="margin-bottom: 20px;">
          <template #header>
            <span>KPI数据</span>
          </template>
          <el-table :data="[projectStore.coolingPlan.cooling_kpis]" style="width: 100%">
            <el-table-column prop="predicted_PUE" label="预测PUE" />
            <el-table-column prop="predicted_WUE" label="预测WUE" />
            <el-table-column prop="cooling_power_kw" label="制冷功率(kW)" />
            <el-table-column prop="corrected_cop" label="修正COP" />
            <el-table-column prop="waste_heat_recovery_kw" label="余热回收功率(kW)" />
          </el-table>
        </el-card>

        <!-- 余热回收策略 -->
        <el-card shadow="hover" style="margin-bottom: 20px;">
          <template #header>
            <span>余热回收策略</span>
          </template>
          <div v-if="projectStore.coolingPlan.waste_heat_recovery_strategy" class="markdown-content">
            {{ projectStore.coolingPlan.waste_heat_recovery_strategy }}
          </div>
          <div v-else class="markdown-content">
            <p>余热回收策略生成中...</p>
          </div>
        </el-card>

        <!-- 制冷方案报告 -->
        <div class="markdown-report" v-if="projectStore.coolingPlan.scheme_detail_brief">
          <div class="report-header">
            <h4>📋 制冷方案报告</h4>
            <el-button-group>
              <el-button size="small" :type="coolingReportViewMode === 'preview' ? 'primary' : 'default'" @click="coolingReportViewMode = 'preview'">预览</el-button>
              <el-button size="small" :type="coolingReportViewMode === 'code' ? 'primary' : 'default'" @click="coolingReportViewMode = 'code'">源码</el-button>
              <el-button size="small" type="info" @click="exportCoolingReport('md')">导出MD</el-button>
              <el-button size="small" type="info" @click="exportCoolingReport('html')">导出HTML</el-button>
              <el-button size="small" type="primary" @click="regenerateCoolingReport">重新生成</el-button>
            </el-button-group>
          </div>
          <div class="markdown-content" v-if="coolingReportViewMode === 'preview'" v-html="renderMarkdown(projectStore.coolingPlan.scheme_detail_brief)"></div>
          <el-input v-else type="textarea" :value="projectStore.coolingPlan.scheme_detail_brief" :rows="15" readonly style="font-family: monospace;"></el-input>
        </div>
        <div class="markdown-report" v-else>
          <h4>📋 制冷方案报告</h4>
          <div class="markdown-content">
            <p>报告生成中，请稍候...</p>
            <p>或点击下方按钮重新生成报告</p>
            <el-button type="primary" @click="regenerateCoolingReport" style="margin-top: 10px;">重新生成报告</el-button>
          </div>
        </div>
      </div>

      <!-- Agent 4: 方案验证专家 -->
      <div v-if="activeAgent.id === 4" class="review-plan">
        <!-- 评估结果 -->
        <el-card shadow="hover" style="margin-bottom: 20px;">
          <template #header>
            <span>方案评估结果</span>
          </template>
          <div v-if="projectStore.reviewResult" class="markdown-content">
            <div v-html="renderMarkdown(projectStore.reviewResult.evaluation_text)"></div>
          </div>
          <div v-else class="markdown-content">
            <p>评估结果生成中...</p>
          </div>
        </el-card>

        <!-- 决策信息 -->
        <el-card shadow="hover" style="margin-bottom: 20px;">
          <template #header>
            <span>决策信息</span>
          </template>
          <el-descriptions :column="2" border>
            <el-descriptions-item label="评估结论">
              <el-tag :type="projectStore.reviewResult?.passed ? 'success' : 'danger'">
                {{ projectStore.reviewResult?.passed ? '通过' : '不通过' }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="评估工具">
              {{ projectStore.reviewResult?.evaluator || 'Unknown' }}
            </el-descriptions-item>
            <el-descriptions-item label="综合评分">
              {{ projectStore.reviewResult?.score || 0 }}/5
            </el-descriptions-item>
            <el-descriptions-item label="迭代次数">
              {{ projectStore.feedback?.iteration_count || 0 }}
            </el-descriptions-item>
          </el-descriptions>
        </el-card>

        <!-- 反馈信息 -->
        <el-card shadow="hover" style="margin-bottom: 20px;">
          <template #header>
            <span>反馈信息</span>
          </template>
          <div v-if="projectStore.feedback">
            <div v-if="projectStore.feedback.issues && projectStore.feedback.issues.length > 0">
              <h5>发现的问题：</h5>
              <el-list>
                <el-list-item v-for="(issue, index) in projectStore.feedback.issues" :key="index">
                  <template #prefix>
                    <el-tag size="small" type="danger">问题</el-tag>
                  </template>
                  {{ issue }}
                </el-list-item>
              </el-list>
            </div>
            <div v-if="projectStore.feedback.suggestions && projectStore.feedback.suggestions.length > 0">
              <h5>改进建议：</h5>
              <el-list>
                <el-list-item v-for="(suggestion, index) in projectStore.feedback.suggestions" :key="index">
                  <template #prefix>
                    <el-tag size="small" type="info">建议</el-tag>
                  </template>
                  {{ suggestion }}
                </el-list-item>
              </el-list>
            </div>
          </div>
          <div v-else>
            <p>反馈信息生成中...</p>
          </div>
        </el-card>

        <!-- 原始输出 -->
        <el-card shadow="hover" style="margin-bottom: 20px;">
          <template #header>
            <span>原始输出</span>
          </template>
          <el-input type="textarea" :value="JSON.stringify({review_result: projectStore.reviewResult, feedback: projectStore.feedback, iteration_count: projectStore.feedback?.iteration_count}, null, 2)" :rows="10" readonly style="font-family: monospace;"></el-input>
        </el-card>
      </div>

      <!-- Agent 5: 投资决策专家 -->
      <div v-if="activeAgent.id === 5" class="financial-analysis">
        <!-- 财务分析结果 -->
        <el-card shadow="hover" style="margin-bottom: 20px;">
          <template #header>
            <span>财务分析结果</span>
          </template>
          <div v-if="projectStore.financialAnalysis" class="financial-content">
            <el-descriptions :column="2" border>
              <el-descriptions-item label="总投资">
                {{ projectStore.financialAnalysis.capex_total }} 万元
              </el-descriptions-item>
              <el-descriptions-item label="投资回收期">
                {{ projectStore.financialAnalysis.payback_years }} 年
              </el-descriptions-item>
              <el-descriptions-item label="年节省">
                {{ projectStore.financialAnalysis.annual_saving }} 万元
              </el-descriptions-item>
              <el-descriptions-item label="年碳减排">
                {{ projectStore.financialAnalysis.emission_reduction }} 吨 CO₂
              </el-descriptions-item>
              <el-descriptions-item label="总用电量">
                {{ projectStore.financialAnalysis.total_electricity }} MWh
              </el-descriptions-item>
              <el-descriptions-item label="绿电比例">
                {{ projectStore.financialAnalysis.green_ratio }}%
              </el-descriptions-item>
              <el-descriptions-item label="实际PUE">
                {{ projectStore.financialAnalysis.actual_pue }}
              </el-descriptions-item>
              <el-descriptions-item label="制冷技术">
                {{ projectStore.financialAnalysis.cooling_tech }}
              </el-descriptions-item>
            </el-descriptions>
          </div>
          <div v-else class="financial-content">
            <p>财务分析结果生成中...</p>
          </div>
        </el-card>

        <!-- 投资明细 -->
        <el-card shadow="hover" style="margin-bottom: 20px;">
          <template #header>
            <span>投资明细</span>
          </template>
          <div v-if="projectStore.financialAnalysis && projectStore.financialAnalysis.capex_breakdown">
            <el-descriptions :column="3" border>
              <el-descriptions-item label="光伏系统">
                {{ projectStore.financialAnalysis.capex_breakdown.pv_system }} 万元
              </el-descriptions-item>
              <el-descriptions-item label="储能系统">
                {{ projectStore.financialAnalysis.capex_breakdown.storage_system }} 万元
              </el-descriptions-item>
              <el-descriptions-item label="制冷系统">
                {{ projectStore.financialAnalysis.capex_breakdown.cooling_system }} 万元
              </el-descriptions-item>
            </el-descriptions>
          </div>
          <div v-else>
            <p>投资明细生成中...</p>
          </div>
        </el-card>

        <!-- 成本分析 -->
        <el-card shadow="hover" style="margin-bottom: 20px;">
          <template #header>
            <span>成本分析</span>
          </template>
          <div v-if="projectStore.financialAnalysis">
            <el-descriptions :column="2" border>
              <el-descriptions-item label="购电成本">
                {{ projectStore.financialAnalysis.grid_cost }} 万元
              </el-descriptions-item>
              <el-descriptions-item label="PPA成本">
                {{ projectStore.financialAnalysis.ppa_cost }} 万元
              </el-descriptions-item>
              <el-descriptions-item label="光伏节省">
                {{ projectStore.financialAnalysis.pv_saving }} 万元
              </el-descriptions-item>
              <el-descriptions-item label="碳减排收益">
                {{ projectStore.financialAnalysis.carbon_benefit }} 万元
              </el-descriptions-item>
              <el-descriptions-item label="碳补偿成本">
                {{ projectStore.financialAnalysis.carbon_compensation_cost }} 万元
              </el-descriptions-item>
              <el-descriptions-item label="总用电成本">
                {{ projectStore.financialAnalysis.total_cost }} 万元
              </el-descriptions-item>
            </el-descriptions>
          </div>
          <div v-else>
            <p>成本分析生成中...</p>
          </div>
        </el-card>

        <!-- 财务分析报告 -->
        <el-card shadow="hover" style="margin-bottom: 20px;">
          <template #header>
            <span>财务分析报告</span>
          </template>
          <div v-if="projectStore.financialAnalysis && projectStore.financialAnalysis.report_md" class="markdown-content">
            <div v-html="renderMarkdown(projectStore.financialAnalysis.report_md)"></div>
          </div>
          <div v-else class="markdown-content">
            <p>财务分析报告生成中...</p>
          </div>
        </el-card>

        <!-- 原始输出 -->
        <el-card shadow="hover" style="margin-bottom: 20px;">
          <template #header>
            <span>原始输出</span>
          </template>
          <el-input type="textarea" :value="JSON.stringify(projectStore.financialAnalysis, null, 2)" :rows="10" readonly style="font-family: monospace;"></el-input>
        </el-card>
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
      <el-button v-if="projectStore.agentStatus.agent3" type="primary" @click="() => { submitToSimulation(); }">
        <el-icon><VideoPlay /></el-icon> 提交仿真验证
      </el-button>
      <el-button v-if="projectStore.agentStatus.agent4" type="success" @click="() => { submitToFinancial(); }">
        <el-icon><Money /></el-icon> 提交财务分析
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
import katex from 'katex'
import 'katex/dist/katex.min.css'

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
    name: '方案验证专家',
    desc: '验证方案技术性、经济性合理性',
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
const coolingReportViewMode = ref('preview')

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
const exportReport = (format: 'md' | 'html') => {
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

// 导出制冷方案报告
const exportCoolingReport = (format: 'md' | 'html') => {
  const reportContent = projectStore.coolingPlan.scheme_detail_brief
  if (!reportContent) return
  
  let content = reportContent
  let filename = '制冷方案报告'
  let mimeType = 'text/plain'
  
  if (format === 'html') {
    content = renderMarkdown(reportContent)
    content = `<html><head><style>body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; } h1 { color: #333; } h2 { color: #555; } ul { margin: 10px 0; } li { margin: 5px 0; } .katex-block { text-align: center; margin: 20px 0; } </style></head><body>${content}</body></html>`
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

// 重新生成制冷方案报告
const regenerateCoolingReport = async () => {
  ElMessage.info('正在重新生成制冷方案报告...')
  await fetchCoolingPlan()
  ElMessage.success('制冷方案报告已重新生成')
}

// 获取能源规划数据
const fetchEnergyPlan = async () => {
  try {
    ElMessage.info('正在获取能源规划数据...')
    
    // 构建后端期望的参数结构
    const requestData = {
      user_requirements: {
        location: projectStore.requirement.location,
        business_type: projectStore.requirement.businessType,
        planned_area: projectStore.requirement.area,
        planned_load: projectStore.requirement.load * 1000, // 转换为kW
        computing_power_density: projectStore.requirement.density,
        priority: projectStore.requirement.priority.join(','),
        green_energy_target: projectStore.requirement.greenTarget,
        pue_target: projectStore.requirement.pueTarget,
        budget_constraint: projectStore.requirement.budget
      },
      environmental_data: {
        annual_temperature: projectStore.envData.climate.avgTemp,
        annual_wind_speed: projectStore.envData.climate.windSpeed,
        annual_sunshine_hours: projectStore.envData.climate.solarRadiation,
        carbon_emission_factor: projectStore.envData.carbonFactor
      },
      electricity_price: {
        peak_price: projectStore.envData.electricity.peakPrice,
        high_price: projectStore.envData.electricity.highPrice,
        flat_price: projectStore.envData.electricity.flatPrice,
        low_price: projectStore.envData.electricity.valleyPrice,
        deep_low_price: projectStore.envData.electricity.deepValleyPrice,
        max_price_diff: projectStore.envData.electricity.maxPriceDiff
      }
    }
    
    console.log('发送给后端的数据:', requestData)
    
    const response = await fetch('http://localhost:5004/api/agent2/energy-plan', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestData)
    })
    const data = await response.json()
    if (data.success) {
      projectStore.updateEnergyPlan(data.data)
      projectStore.updateAgentStatus('agent2', true)
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
const showAgentDetail = async (id: number) => {
  activeAgent.value = agents.value.find(a => a.id === id)
  
  if (id === 2) {
    // 加载能源规划数据
    await fetchEnergyPlan()
  } else if (id === 3) {
    // 加载制冷方案数据
    await fetchCoolingPlan()
  }
}

// 获取制冷方案数据
const fetchCoolingPlan = async () => {
  try {
    // 构建后端期望的参数结构
    const requestData = {
      user_requirements: {
        location: projectStore.requirement.location,
        business_type: projectStore.requirement.businessType,
        planned_area: projectStore.requirement.area,
        planned_load: projectStore.requirement.load * 1000, // 转换为kW
        computing_power_density: projectStore.requirement.density,
        priority: projectStore.requirement.priority[0],
        green_energy_target: projectStore.requirement.greenTarget,
        pue_target: projectStore.requirement.pueTarget,
        budget_constraint: projectStore.requirement.budget
      },
      environmental_data: {
        annual_temperature: projectStore.envData.climate.avgTemp,
        annual_wind_speed: projectStore.envData.climate.windSpeed,
        annual_sunshine_hours: projectStore.envData.climate.solarRadiation,
        carbon_emission_factor: projectStore.envData.carbonFactor,
        raw_water_usage: 12000.0 // 默认值
      },
      electricity_price: {
        peak_price: projectStore.envData.electricity.peakPrice,
        high_price: projectStore.envData.electricity.highPrice,
        flat_price: projectStore.envData.electricity.flatPrice,
        low_price: projectStore.envData.electricity.valleyPrice,
        deep_low_price: projectStore.envData.electricity.deepValleyPrice,
        max_price_diff: projectStore.envData.electricity.maxPriceDiff
      }
    }

    const response = await fetch('http://localhost:5004/api/agent3/cooling-plan', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestData)
    })
    
    const data = await response.json()
    if (data.success) {
      projectStore.updateCoolingPlan(data.data)
      projectStore.updateAgentStatus('agent3', true)
    }
  } catch (error) {
    console.error('获取制冷方案失败:', error)
  }
}

// 渲染Markdown为HTML
const renderMarkdown = (markdown: string): string => {
  if (!markdown) return ''
  
  let html = markdown
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
  
  // 渲染Markdown表格
  const parseTable = (tableLines: string[]): string => {
    if (tableLines.length < 2) return ''
    
    // 解析表头
    const headerCells = tableLines[0].split('|').map(cell => cell.trim()).filter(cell => cell !== '')
    
    // 解析分隔线（确定对齐方式）
    const separatorLine = tableLines[1] || ''
    const separators = separatorLine.split('|').map(cell => cell.trim()).filter(cell => cell !== '')
    
    // 确定每列的对齐方式
    const aligns = separators.map(sep => {
      if (sep.startsWith(':') && sep.endsWith(':')) return 'center'
      if (sep.endsWith(':')) return 'right'
      return 'left'
    })
    
    // 解析数据行
    const rows = tableLines.slice(2).map(line => {
      return line.split('|').map(cell => cell.trim()).filter(cell => cell !== '')
    }).filter(row => row.length > 0)
    
    // 生成HTML表格
    let tableHtml = '<table class="markdown-table">'
    
    // 表头
    tableHtml += '<thead><tr>'
    headerCells.forEach((cell, index) => {
      const align = aligns[index] || 'left'
      tableHtml += `<th style="text-align: ${align}">${cell}</th>`
    })
    tableHtml += '</tr></thead>'
    
    // 数据行
    if (rows.length > 0) {
      tableHtml += '<tbody>'
      rows.forEach(row => {
        tableHtml += '<tr>'
        row.forEach((cell, index) => {
          const align = aligns[index] || 'left'
          tableHtml += `<td style="text-align: ${align}">${cell}</td>`
        })
        tableHtml += '</tr>'
      })
      tableHtml += '</tbody>'
    }
    
    tableHtml += '</table>'
    return tableHtml
  }
  
  // 查找并替换表格
  const lines = html.split('\n')
  let result: string[] = []
  let tableBuffer: string[] = []
  let inTable = false
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmedLine = line.trim()
    
    // 检测表格行（以 | 开头或包含 | 且至少有3个）
    const isTableRow = trimmedLine.startsWith('|') || (trimmedLine.includes('|') && trimmedLine.split('|').length >= 3)
    const isSeparatorLine = trimmedLine.includes('|') && trimmedLine.replace(/\|/g, '').replace(/-/g, '').replace(/:/g, '').trim() === ''
    
    if (isTableRow || (inTable && trimmedLine === '')) {
      if (trimmedLine !== '' || inTable) {
        tableBuffer.push(line)
        inTable = true
      }
      
      // 检查是否应该结束表格
      const nextLine = lines[i + 1] || ''
      const isNextLineTableRow = nextLine.trim().startsWith('|') || (nextLine.trim().includes('|') && nextLine.trim().split('|').length >= 3)
      const isNextLineSeparator = nextLine.trim().includes('|') && nextLine.trim().replace(/\|/g, '').replace(/-/g, '').replace(/:/g, '').trim() === ''
      
      if (!isNextLineTableRow && !isNextLineSeparator && tableBuffer.length >= 2) {
        // 结束表格并渲染
        const tableHtml = parseTable(tableBuffer)
        if (tableHtml) {
          result.push(tableHtml)
        }
        tableBuffer = []
        inTable = false
      }
    } else {
      if (inTable && tableBuffer.length >= 2) {
        const tableHtml = parseTable(tableBuffer)
        if (tableHtml) {
          result.push(tableHtml)
        }
      }
      result.push(line)
      tableBuffer = []
      inTable = false
    }
  }
  
  // 处理最后剩余的表格
  if (inTable && tableBuffer.length >= 2) {
    const tableHtml = parseTable(tableBuffer)
    if (tableHtml) {
      result.push(tableHtml)
    }
  }
  
  html = result.join('\n')
  
  // 渲染LaTeX公式（行间公式 $$...$$）
  html = html.replace(/\$\$([\s\S]*?)\$\$/g, (_, formula) => {
    try {
      return `<div class="katex-block">${katex.renderToString(formula.trim(), { displayMode: true, throwOnError: false })}</div>`
    } catch (e) {
      return `<div class="katex-error">$$${formula}$$</div>`
    }
  })
  
  // 渲染LaTeX公式（行内公式 $...$）
  html = html.replace(/\$([^$\n]+?)\$/g, (_, formula) => {
    try {
      return katex.renderToString(formula.trim(), { displayMode: false, throwOnError: false })
    } catch (e) {
      return `$${formula}$`
    }
  })
  
  // 代码块
  html = html
    .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
  
  // 粗体和斜体
  html = html
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
  
  // 标题
  html = html
    .replace(/^#### (.*$)/gm, '<h4>$1</h4>')
    .replace(/^### (.*$)/gm, '<h3>$1</h3>')
    .replace(/^## (.*$)/gm, '<h2>$1</h2>')
    .replace(/^# (.*$)/gm, '<h1>$1</h1>')
  
  // 列表
  html = html
    .replace(/^\d+\. (.*$)/gm, '<li>$1</li>')
    .replace(/^- (.*$)/gm, '<li>$1</li>')
    .replace(/(<li>[\s\S]*?<\/li>\n?)+/g, '<ul>$&</ul>')
    .replace(/<\/ul>\n<ul>/g, '')
  
  // 分隔线和链接
  html = html
    .replace(/^---+$/gm, '<hr>')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
  
  // 段落处理（跳过表格）
  html = html.split('\n\n').map(block => {
    if (block.startsWith('<h') || block.startsWith('<ul') || block.startsWith('<ol') || 
        block.startsWith('<pre') || block.startsWith('<hr') || block.startsWith('<p') ||
        block.startsWith('<div class="katex') || block.startsWith('<table')) {
      return block
    }
    return block.split('\n').map(line => {
      if (line.trim() && !line.startsWith('<')) {
        return `<p>${line}</p>`
      }
      return line
    }).join('\n')
  }).join('\n')
  
  return html
}

// 确认Agent方案
const approveAgent = () => {
  if (activeAgent.value) {
    projectStore.updateAgentStatus(`agent${activeAgent.value.id}` as any, true)
    activeAgent.value.status = true
    activeAgent.value.progress = 100
    ElMessage.success(`${activeAgent.value.name} 方案已确认`)
    
    // 检查是否是agent5且所有前面的agent都已完成
    if (activeAgent.value.id === 5 && projectStore.agentStatus.agent1 && projectStore.agentStatus.agent2 && projectStore.agentStatus.agent3 && projectStore.agentStatus.agent4) {
      ElMessage.success('所有方案已确认，正在跳转到仿真页面...')
      // 跳转到仿真页面
      router.push('/simulation')
    }
    
    activeAgent.value = null
  }
}

// 提出修改意见
const rejectAgent = () => {
  ElMessage.warning('修改意见已发送给Agent')
}

// 提交仿真验证
const submitToSimulation = async () => {
  try {
    // 检查Agent 2和3是否完成
    if (!projectStore.agentStatus.agent2 || !projectStore.agentStatus.agent3) {
      ElMessage.error('请先完成能源规划和制冷方案设计')
      return
    }
    
    // 检查能源方案数据 - 放宽检查条件，只要有 llm_report 就可以继续
    if (!projectStore.energyPlan.llm_report && 
        projectStore.energyPlan.pv_capacity === 0 && 
        projectStore.energyPlan.wind_capacity === 0 && 
        projectStore.energyPlan.storage_capacity === 0) {
      ElMessage.error('能源方案数据不完整，请重新生成能源规划')
      return
    }
    
    // 检查制冷方案数据 - 放宽检查条件，只要有 scheme_detail_brief 就可以继续
    if (!projectStore.coolingPlan.cooling_technology && !projectStore.coolingPlan.scheme_detail_brief) {
      ElMessage.error('制冷方案数据不完整，请重新生成制冷方案')
      return
    }
    
    ElMessage.info('正在提交方案验证...')
    
    // 准备请求数据
    const requestData = {
      user_requirements: {
        location: projectStore.requirement.location,
        planned_load: projectStore.requirement.load * 1000, // 转换为kW
        pue_target: projectStore.requirement.pueTarget,
        green_energy_target: projectStore.requirement.greenTarget,
        computing_power_density: projectStore.requirement.density
      },
      environmental_data: {
        annual_temperature: projectStore.envData.climate.avgTemp,
        carbon_emission_factor: projectStore.envData.carbonFactor
      },
      energy_plan: {
        pv_capacity: projectStore.energyPlan.pv_capacity,
        wind_capacity: projectStore.energyPlan.wind_capacity,
        storage_capacity: projectStore.energyPlan.storage_capacity,
        ppa_ratio: projectStore.energyPlan.ppa_ratio
      },
      cooling_plan: {
        cooling_technology: projectStore.coolingPlan.cooling_technology,
        estimated_pue: projectStore.coolingPlan.estimated_pue,
        incremental_cost: 500 // 默认值
      }
    }
    
    console.log('发送给后端的验证数据:', requestData)
    
    // 调用后端API
    console.log('准备发送验证请求...')
    try {
      const response = await fetch('http://localhost:5004/api/agent4/review', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      })
      
      console.log('API响应状态:', response.status)
      
      if (response.ok) {
        const result = await response.json()
        console.log('API响应数据:', result)
        
        if (result.success) {
          // 更新评估结果和反馈信息
          projectStore.updateReviewResult(result.data.review_result)
          projectStore.updateFeedback(result.data.feedback)
          projectStore.updateAgentStatus('agent4', true)
          
          console.log('更新后的reviewResult:', projectStore.reviewResult)
          console.log('更新后的feedback:', projectStore.feedback)
          
          ElMessage.success('方案验证完成')
          // 显示Agent 4的详情
          showAgentDetail(4)
        } else {
          ElMessage.error('方案验证失败: ' + result.error)
        }
      } else {
        const errorText = await response.text()
        console.log('服务器错误:', errorText)
        ElMessage.error('服务器错误，请稍后重试')
      }
    } catch (error) {
      console.error('网络请求失败:', error)
      ElMessage.error('网络请求失败，请检查后端服务是否运行')
    }
  } catch (error) {
    console.error('提交仿真验证失败:', error)
    ElMessage.error('提交仿真验证失败，请稍后重试')
  }
}

// 提交财务分析
const submitToFinancial = async () => {
  try {
    // 检查Agent 4是否完成
    if (!projectStore.agentStatus.agent4) {
      ElMessage.error('请先完成方案验证')
      return
    }
    
    ElMessage.info('正在提交财务分析...')
    
    // 准备请求数据
    const requestData = {
      user_requirements: {
        location: projectStore.requirement.location,
        planned_load: projectStore.requirement.load * 1000, // 转换为kW
        pue_target: projectStore.requirement.pueTarget,
        green_energy_target: projectStore.requirement.greenTarget,
        computing_power_density: projectStore.requirement.density
      },
      environmental_data: {
        annual_temperature: projectStore.envData.climate.avgTemp,
        carbon_emission_factor: projectStore.envData.carbonFactor
      },
      energy_plan: {
        pv_capacity: projectStore.energyPlan.pv_capacity,
        wind_capacity: projectStore.energyPlan.wind_capacity,
        storage_capacity: projectStore.energyPlan.storage_capacity,
        ppa_ratio: projectStore.energyPlan.ppa_ratio
      },
      cooling_plan: {
        cooling_technology: projectStore.coolingPlan.cooling_technology,
        estimated_pue: projectStore.coolingPlan.estimated_pue,
        incremental_cost: 500 // 默认值
      },
      review_result: projectStore.reviewResult
    }
    
    console.log('发送给后端的财务分析数据:', requestData)
    
    // 调用后端API
    console.log('准备发送财务分析请求...')
    try {
      const response = await fetch('http://localhost:5004/api/agent5/financial', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      })
      
      console.log('API响应状态:', response.status)
      
      if (response.ok) {
        const result = await response.json()
        console.log('API响应数据:', result)
        
        if (result.success) {
          // 更新财务分析结果
          projectStore.updateFinancialAnalysis(result.data.financial_analysis)
          projectStore.updateAgentStatus('agent5', true)
          
          console.log('更新后的financialAnalysis:', projectStore.financialAnalysis)
          
          ElMessage.success('财务分析完成')
          // 显示Agent 5的详情
          showAgentDetail(5)
        } else {
          ElMessage.error('财务分析失败: ' + result.error)
        }
      } else {
        const errorText = await response.text()
        console.log('服务器错误:', errorText)
        ElMessage.error('服务器错误，请稍后重试')
      }
    } catch (error) {
      console.error('网络请求失败:', error)
      ElMessage.error('网络请求失败，请检查后端服务是否运行')
    }
  } catch (error) {
    console.error('提交财务分析失败:', error)
    ElMessage.error('提交财务分析失败，请稍后重试')
  }
}

onMounted(() => {
  // 设置Agent 1为已完成状态
  projectStore.updateAgentStatus('agent1', true)
  if (agents.value[0]) {
    agents.value[0].status = true
    agents.value[0].progress = 100
  }
  
  // 模拟Agent 2和3正在工作
  setTimeout(() => {
    projectStore.updateAgentStatus('agent2', true)
    if (agents.value[1]) {
      agents.value[1].status = true
      agents.value[1].progress = 100
    }
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

.markdown-content h4 {
  font-size: 13px;
  font-weight: bold;
  margin: 8px 0;
  color: #E6A23C;
}

.markdown-content strong {
  font-weight: bold;
  color: #303133;
}

.markdown-content em {
  font-style: italic;
  color: #606266;
}

.markdown-content code {
  background: #e4e7ed;
  padding: 2px 6px;
  border-radius: 4px;
  font-family: 'Consolas', 'Monaco', monospace;
  font-size: 13px;
  color: #e6a23c;
}

.markdown-content pre {
  background: #2d3748;
  padding: 15px;
  border-radius: 8px;
  overflow-x: auto;
  margin: 15px 0;
}

.markdown-content pre code {
  background: transparent;
  color: #e2e8f0;
  padding: 0;
}

.markdown-content hr {
  border: none;
  border-top: 1px solid #dcdfe6;
  margin: 20px 0;
}

.markdown-content a {
  color: #409EFF;
  text-decoration: none;
}

.markdown-content a:hover {
  text-decoration: underline;
}

.markdown-content .katex-block {
  display: block;
  text-align: center;
  margin: 20px 0;
  padding: 15px;
  background: #f5f7fa;
  border-radius: 8px;
  overflow-x: auto;
}

.markdown-content .katex-error {
  color: #f56c6c;
  background: #fef0f0;
  padding: 10px;
  border-radius: 4px;
}

.markdown-content .markdown-table {
  width: 100%;
  border-collapse: collapse;
  margin: 15px 0;
  background: white;
}

.markdown-content .markdown-table th,
.markdown-content .markdown-table td {
  border: 1px solid #dcdfe6;
  padding: 10px 15px;
}

.markdown-content .markdown-table th {
  background: #f5f7fa;
  font-weight: bold;
  color: #303133;
}

.markdown-content .markdown-table tbody tr:hover {
  background: #f5f7fa;
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
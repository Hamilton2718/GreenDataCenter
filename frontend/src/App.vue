<template>
  <el-container class="app-container">
    <el-header class="app-header">
      <div class="header-left">
        <h2>🌱 数据中心绿电消纳规划设计顾问</h2>
      </div>
      <div class="header-right">
        <el-button type="primary" link>
          <el-icon><User /></el-icon>
          管理员
        </el-button>
        <el-button type="primary" link>
          <el-icon><Download /></el-icon>
          导出
        </el-button>
        <el-button type="primary" link>
          <el-icon><Share /></el-icon>
          分享
        </el-button>
      </div>
    </el-header>
    
    <el-container>
      <el-aside width="200px" class="app-aside">
        <el-menu
          :router="true"
          :default-active="activeMenu"
          class="side-menu"
        >
          <el-menu-item index="/requirement">
            <el-icon><Document /></el-icon>
            <span>1. 需求输入</span>
          </el-menu-item>
          <el-menu-item index="/design">
            <el-icon><Setting /></el-icon>
            <span>2. 方案设计</span>
          </el-menu-item>
          <el-menu-item index="/simulation">
            <el-icon><VideoPlay /></el-icon>
            <span>3. 仿真运行</span>
          </el-menu-item>
          <el-menu-item index="/report">
            <el-icon><DataLine /></el-icon>
            <span>4. 结果输出</span>
          </el-menu-item>
        </el-menu>
      </el-aside>
      
      <el-main class="app-main">
        <router-view v-slot="{ Component }">
          <transition name="fade" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </el-main>
    </el-container>
    
    <el-footer class="app-footer">
      <div class="agent-status-bar">
        <div class="status-item" :class="{ active: agentStatus.agent1 }">
          <el-icon><Checked /></el-icon>
          Agent 1: 需求解析
          <span class="status-badge" :class="agentStatus.agent1 ? 'success' : 'waiting'">
            {{ agentStatus.agent1 ? '✅完成' : '⏳等待' }}
          </span>
        </div>
        <div class="status-item" :class="{ active: agentStatus.agent2 }">
          <el-icon><Checked /></el-icon>
          Agent 2: 能源规划
          <span class="status-badge" :class="agentStatus.agent2 ? 'success' : 'waiting'">
            {{ agentStatus.agent2 ? '✅完成' : '⏳等待' }}
          </span>
        </div>
        <div class="status-item" :class="{ active: agentStatus.agent3 }">
          <el-icon><Checked /></el-icon>
          Agent 3: 制冷设计
          <span class="status-badge" :class="agentStatus.agent3 ? 'success' : 'waiting'">
            {{ agentStatus.agent3 ? '✅完成' : '⏳等待' }}
          </span>
        </div>
        <div class="status-item" :class="{ active: agentStatus.agent4 }">
          <el-icon><Checked /></el-icon>
          Agent 4: 仿真验证
          <span class="status-badge" :class="agentStatus.agent4 ? 'success' : 'waiting'">
            {{ agentStatus.agent4 ? '✅完成' : '⏳等待' }}
          </span>
        </div>
        <div class="status-item" :class="{ active: agentStatus.agent5 }">
          <el-icon><Checked /></el-icon>
          Agent 5: 投资决策
          <span class="status-badge" :class="agentStatus.agent5 ? 'success' : 'waiting'">
            {{ agentStatus.agent5 ? '✅完成' : '⏳等待' }}
          </span>
        </div>
      </div>
    </el-footer>
  </el-container>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useProjectStore } from './stores/projectStore'

const route = useRoute()
const projectStore = useProjectStore()

const activeMenu = computed(() => route.path)

// Agent状态（从store获取）
const agentStatus = computed(() => projectStore.agentStatus)
</script>

<style scoped>
.app-container {
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-header {
  background-color: #2c3e50;
  color: white;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.header-left h2 {
  margin: 0;
  font-size: 1.5rem;
}

.header-right .el-button {
  color: white;
  margin-left: 16px;
}

.app-aside {
  background-color: #f5f7fa;
  border-right: 1px solid #e4e7ed;
}

.side-menu {
  border-right: none;
  height: 100%;
}

.app-main {
  background-color: #f0f2f5;
  padding: 24px;
  overflow-y: auto;
}

.app-footer {
  background-color: white;
  border-top: 1px solid #e4e7ed;
  padding: 12px 24px;
  height: auto;
}

.agent-status-bar {
  display: flex;
  justify-content: space-around;
  align-items: center;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border-radius: 20px;
  background-color: #f5f7fa;
  color: #909399;
}

.status-item.active {
  background-color: #ecf5ff;
  color: #409eff;
}

.status-badge {
  font-size: 0.85rem;
  padding: 2px 8px;
  border-radius: 12px;
}

.status-badge.success {
  background-color: #67c23a;
  color: white;
}

.status-badge.waiting {
  background-color: #909399;
  color: white;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
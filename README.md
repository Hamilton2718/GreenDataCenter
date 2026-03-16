# 绿色数据中心规划系统

## 项目结构

```
GreenDataCenter/
├── frontend/           # 前端Vue项目
│   ├── public/         # 静态资源
│   ├── src/            # 源代码
│   │   ├── assets/     # 资源文件
│   │   ├── components/ # 组件
│   │   ├── router/     # 路由
│   │   ├── stores/     # 状态管理
│   │   ├── views/      # 页面
│   │   ├── App.vue     # 根组件
│   │   └── main.ts     # 入口文件
│   ├── index.html      # HTML模板
│   ├── package.json    # 依赖配置
│   └── vite.config.ts  # Vite配置
├── nodes/              # 智能体节点
│   ├── requirement_analysis_node.py  # 需求分析节点
│   └── energy_planner_node.py        # 能源规划节点
├── app.py              # 后端Flask应用
├── requirements.txt    # Python依赖
└── README.md           # 项目说明
```

## 运行说明

### 后端服务

1. 进入项目目录
   ```bash
   cd GreenDataCenter
   ```

2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

3. 启动后端服务
   ```bash
   python app.py
   ```
   服务将运行在 http://localhost:5001

### 前端服务

1. 进入前端目录
   ```bash
   cd GreenDataCenter/frontend
   ```

2. 安装依赖
   ```bash
   npm install
   ```

3. 启动前端开发服务
   ```bash
   npm run dev
   ```
   服务将运行在 http://localhost:5173

## 功能说明

### 核心功能

1. **需求解析**：分析用户输入的项目需求，包括地理位置、业务类型、计划面积、计划负荷等
2. **环境分析**：基于地理位置获取环境数据，包括温度、风速、日照时数等
3. **能源规划**：制定能源规划方案，包括光伏、风电、储能、绿电长协等配置
4. **制冷设计**：设计制冷方案，计算PUE
5. **仿真验证**：模拟24小时运行情况，验证方案可行性
6. **投资决策**：评估成本和投资回报

### 数据关系

- **总耗电量** = 自建绿电（光伏+风电）+ 外部绿电（PPA长协）+ 传统能源（电网调峰）
- **容量配置比例之和** = 光伏占比 + 风电占比 + PPA占比 + 电网占比 = 100%
- **绿电占比** = (自建绿电消纳量 + PPA购电量) / 总用电量 × 100%
- **自发自用率** = 自建绿电实际消纳量 / 自建绿电发电量 × 100%

### API接口

- **POST /api/analyze**：分析项目需求
- **POST /api/stream/analyze**：流式返回分析结果
- **POST /api/agent2/energy-plan**：生成能源规划方案

## 技术栈

- **前端**：Vue 3 + TypeScript + Element Plus + ECharts
- **后端**：Python + Flask + CORS
- **状态管理**：Pinia
- **构建工具**：Vite
- **数据可视化**：ECharts

## 注意事项

1. 确保后端服务在前端服务之前启动
2. 前端服务通过代理访问后端API，代理配置在 vite.config.ts 中
3. 若修改后端端口，需同步修改前端代理配置
4. 首次运行时，后端会下载必要的模型和数据，可能需要一些时间

## 开发指南

### 前端开发

1. 前端代码位于 `frontend/src` 目录
2. 页面组件位于 `frontend/src/views` 目录
3. 状态管理位于 `frontend/src/stores` 目录
4. 路由配置位于 `frontend/src/router` 目录

### 后端开发

1. 后端代码位于 `app.py` 文件
2. 智能体节点位于 `nodes` 目录
3. 新增智能体节点需在 `app.py` 中注册

## 故障排查

1. **后端服务启动失败**：检查端口是否被占用，依赖是否安装完整
2. **前端无法访问后端API**：检查后端服务是否运行，代理配置是否正确
3. **数据显示异常**：检查前后端数据格式是否一致，API返回数据是否正确
4. **能源规划报告不显示**：检查energy_planner_node.py是否正确生成LLM报告

## 版本信息

- 后端API版本：v1.0
- 前端版本：v1.0
- 最后更新时间：2026-03-16

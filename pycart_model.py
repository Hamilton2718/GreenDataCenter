import pandas as pd
import os
from pycaret.time_series import *

# --- 0. 预检环境 ---
# 如果还是报 LightGBM 找不到，请在终端执行: pip install lightgbm
# 或者取消下面这一行的注释运行一次：
# os.system('pip install lightgbm')

# 1. 加载数据
try:
    df = pd.read_csv('power_data.csv', encoding='gbk')
except UnicodeDecodeError:
    df = pd.read_csv('power_data.csv', encoding='utf-8')

# 清理列名：只保留时间和出力两列
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
if df.shape[1] >= 2:
    # 强制取最后两列（通常第一列是时间，第二列是功率）
    df = df.iloc[:, -2:]
    df.columns = ['hour', 'power']

# 转换时间格式
df['hour'] = pd.to_datetime(df['hour'])
df.set_index('hour', inplace=True)

# 显式设置小时频率 (1h)
# 这一步非常重要，处理 1h 数据必须确保索引连续
df = df.asfreq('H')
df['power'] = df['power'].fillna(method='ffill') # 填充可能存在的断档

print("数据准备就绪，形状为:", df.shape)

# 2. 初始化环境
# fh=24 表示预测未来 24 小时
# n_jobs=-1 开启多核并行加速
s = setup(data=df, target='power', fh=24, session_id=123, n_jobs=-1)

# 3. 创建并训练 LightGBM 模型
# 如果环境里确实装不上 lightgbm，请把下面一行改为 model = create_model('rf')
try:
    print("正在训练 LightGBM 模型...")
    model = create_model('lightgbm')
except ValueError:
    print("检测到环境缺少 LightGBM，自动切换为随机森林(rf)...")
    model = create_model('rf')

# 4. 预测验证 (查看最近 24 小时的预测表现)
predictions = predict_model(model)
print("\n测试集预测结果样例:")
print(predictions.head())

# 5. 固化并保存最终模型
# 注意：这里必须传入刚刚创建的 model 对象
final_model = finalize_model(model)

# 保存为 .pkl 文件
save_model(final_model, 'my_power_model_1h')

print("\n--- 训练任务完成 ---")
print("模型已保存为: my_power_model_1h.pkl")
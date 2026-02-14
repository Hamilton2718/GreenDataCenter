import pandas as pd

try:
    df = pd.read_csv('power_data.csv', encoding='gbk')
    print(f"Encoding: gbk, Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"GBK failed: {e}")

try:
    df = pd.read_csv('power_data.csv', encoding='utf-8')
    print(f"Encoding: utf-8, Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"UTF-8 failed: {e}")

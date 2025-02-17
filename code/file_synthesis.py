import pandas as pd
import glob
import os

# 定义输出目录和文件名
output_dir = '../output/synthesis'
output_filename = 'result.csv'

# 确保输出目录存在，如果不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 使用glob模块查找所有分割的CSV文件
files = glob.glob(os.path.join(output_dir, 'result_*.csv'))

# 初始化一个空的DataFrame来存储合并后的数据
full_data = pd.DataFrame()

# 依次读取每个文件并合并
for file in sorted(files):
    # 读取CSV文件
    chunk = pd.read_csv(file, header=None)
    # 将读取的数据追加到full_data DataFrame中
    full_data = pd.concat([full_data, chunk], ignore_index=True)

# 保存合并后的数据到原始文件
full_data.to_csv(os.path.join(output_dir, output_filename), index=False, header=False)
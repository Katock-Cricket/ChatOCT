import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# JSON文件路径列表，填入你的文件名
eval_list = [
    # 'eval/report_length/result_gpt3.5.json',
    # 'eval/report_length/result_gpt4.json',
    'eval/report_length/result_gpt3.5_retrieval.json',
    'eval/report_length/result_gpt4_retrieval.json',
    'eval/report_length/result_Huatuo2.json',
    'eval/report_length/result_DISC.json',
]

# 准备图表
plt.figure(figsize=(10, 6))

# 循环处理每个文件
for llm in eval_list:
    # 读取JSON数据
    with open(llm, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 提取所有report_length
    # y_length = [item['report_length'] for item in data['results']]
    y_length = [item['first_answer_length'] for item in data['results']]

    # 使用高斯核密度估计生成平滑曲线
    density = gaussian_kde(y_length)
    density.covariance_factor = lambda: .60  # 控制平滑度
    density._compute_covariance()

    # 设置数据范围和密度
    x = np.linspace(min(y_length), max(y_length), 1000)
    y = density(x)

    # 绘制平滑的密度曲线
    plt.plot(x, y, label=data['engine'])  # 添加标签以区分不同的文件
    plt.fill_between(x, 0, y, alpha=0.1)  # 填充曲线下方的空间

# 添加图例
plt.legend()

# 设置图表标题和轴标签
plt.title('Distribution of Answer Lengths Across Multiple LLMs')
plt.xlabel('Report Length')
plt.ylabel('Density')

# 显示图表
plt.show()

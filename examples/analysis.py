#%%
import os
import sys
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm

sns.reset_defaults()

#%%
# 设置全局字体为 Noto Sans CJK
plt.rcParams['font.family'] = 'sans-serif'

mac_chinese_fonts = [
    'PingFang SC',     # macOS 默认中文字体
    'STHeiti',         # 华文黑体
    'Hiragino Sans GB', # 冬青黑体
    'Noto Sans CJK SC'  # 如果您已安装
]

# 或者使用以下方式也可以
matplotlib.rcParams['font.sans-serif'] = mac_chinese_fonts
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 测试中文显示
x = np.linspace(0, 2*np.pi, 100)
plt.figure(figsize=(10, 6))
plt.plot(x, np.sin(x))
plt.title('正弦函数曲线')
plt.xlabel('角度值 (弧度)')
plt.ylabel('振幅')
plt.show()
#%%
ROOT_DIR = "/Users/floyed/Documents/workplace/jailbreak-pipeline"
JBB_DIR = os.path.join(ROOT_DIR, "benchmarks", "jbb_judged")
DATA_DIR = os.path.join(ROOT_DIR, "data")
#%%
jailbreak_data = pd.read_csv(os.path.join(DATA_DIR, "jbb_expanded.csv"))

goal2category = {}

for i, row in jailbreak_data.iterrows():
    goal2category[row["Goal"]] = row["Category"]

categories = []
categories.extend(list(set(goal2category.values())))

category_count = {}
for k, v in goal2category.items():
    if v not in category_count:
        category_count[v] = 0
    category_count[v] += 1

categories
#%%
# dataframe w/ (model_name, attack_method, ASR)

judge_cls = {
    "GCG": 0,
    "PAIR_gpt-4o-2024-11-20": 0,
    "PAIR_Qwen_Qwen2.5-72B-Instruct": 0,
    "PAIR_meta-llama_Llama-3.3-70B-Instruct": 0,
}

# judge_columns = [f"{goal_cat}_{judge_name}" for goal_cat in categories for judge_name in judge_cls]

df = pd.DataFrame(
    columns=[
        "model_name",
        "attack_method",
        "jailbreak_type",
        *judge_cls.keys(),
    ]
)

#%%

# for root, dirs, files in os.walk(JBB_DIR):
#     if "NoneDefender" not in root:
#         continue
#     for file in files:
#         if file.endswith(".json"):
#             with open(os.path.join(root, file), "r") as f:
#                 model_name = root.split("/")[-3]
#                 if model_name.endswith("_"):
#                     model_name = model_name[:-1]
#                 elif model_name.startswith("aihubmix-"):
#                     model_name = model_name[9:]
#                 elif model_name.startswith("ahm-"):
#                     model_name = model_name[4:]
#                 attack_method = '_'.join(root.split("/")[-2].split("_")[1:])
#                 defense_method = root.split("/")[-1]
#
#                 asr = {k: deepcopy(judge_cls) for k in categories}
#
#                 results = json.load(f)["results"]
#                 for res in results:
#                     ratio = res["jailbroken"]
#                     goal = res["goal"]
#                     for k, v in ratio.items():
#                         if k == 'PAIR_Llama-3.3-70B-Instruct':
#                             k = 'PAIR_meta-llama_Llama-3.3-70B-Instruct'
#                         asr[goal2category[goal]][k] += (v == 10)
#                         # asr[f"{goal2category[goal]}_{k}"] += (v == 10)
#                         # asr[f"All_{k}"] += (v == 10)
#
#                 # asr = {k: v * 1. if "All" in k else v * 10. for k, v in asr.items()}
#
#                 for goal_cat in categories:
#                     asr[goal_cat] = {k: v * 10. for k, v in asr[goal_cat].items()}
#                     new_row = pd.DataFrame([{
#                         "model_name": model_name,
#                         "attack_method": attack_method,
#                         "defense_method": defense_method,
#                         "jailbreak_type": goal_cat,
#                         **asr[goal_cat]
#                     }])
#                     # new_row = new_row.dropna(axis=1, how='all')
#                     df = pd.concat([df, new_row], ignore_index=True)
#%%
# df.to_csv('analysis.csv')

df = pd.read_csv("analysis.csv")
#%%
df['attack_method'].unique()
#%%
df = df[df['model_name'] != 'Llama-3.1-70B-Instruct']
df = df[df['model_name'] != 'Llama-3.3-70B-Instruct']
df['attack_method'] = df['attack_method'].str.replace('new_', '')
print(sorted(df['model_name'].unique()))
#%%
type_mapping = {
    'Physical harm': '身体伤害',
    'Government decision-making': '政府决策',
    'Privacy': '隐私',
    'Malware/Hacking': '恶意软件/黑客攻击',
    'Disinformation': '虚假信息',
    'Sexual/Adult content': '性/成人内容',
    'Harassment/Discrimination': '骚扰/歧视',
    'Expert advice': '专家建议',
    'Economic harm': '经济损害',
    'Fraud/Deception': '欺诈/欺骗'
}

# 使用 replace 方法替换列中的内容
df['jailbreak_type'] = df['jailbreak_type'].replace(type_mapping)


df
#%% md
### 1. Overview Analysis
#%%
numeric_columns = df.select_dtypes(include=["number"]).columns
df_avg = df.groupby("model_name")[numeric_columns].mean()
#%%
df_avg
#%%

for metric in judge_cls:
    # metric = "All_PAIR_gpt-4o-2024-11-20"
    # metric = f"{metric}"

    df_sorted = df_avg.sort_values(by=metric, ascending=True)

    df_sorted = df_sorted.reset_index()

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=df_sorted, y="model_name", x=metric, palette="Blues_d", hue="model_name")

    for p in ax.patches:
        width = p.get_width()  # 获取柱子的宽度（即 ASR 数值）
        ax.text(
            width + 0.01,  # 设置文本的水平位置，稍微偏离柱子的右侧
            p.get_y() + p.get_height() / 2,  # 设置文本的垂直位置，居中于柱子
            f'{width:.1f}%',  # 显示 ASR 值，保留两位小数
            ha='left',  # 水平对齐方式
            va='center',  # 垂直对齐方式
            fontsize=11,  # 字体大小
            color='black'  # 文字颜色
        )

    plt.title(f"攻击成功率 (ASR) (%)", fontsize=16)
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=45, ha='right')

    sns.despine()

    os.makedirs("figures_new/overview", exist_ok=True)
    print(metric)
    plt.savefig(f"figures_new/overview/ASR_{metric}.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()
#%%
df_long = df.melt(id_vars=['model_name', 'attack_method', 'jailbreak_type'],
                  value_vars=['GCG', 'PAIR_gpt-4o-2024-11-20', 'PAIR_Qwen_Qwen2.5-72B-Instruct', 'PAIR_meta-llama_Llama-3.3-70B-Instruct'],
                  var_name='judge_type', value_name='ASR')

# 绘制堆积柱状图
for judge_type in ['GCG', 'PAIR_gpt-4o-2024-11-20', 'PAIR_Qwen_Qwen2.5-72B-Instruct', 'PAIR_meta-llama_Llama-3.3-70B-Instruct']:
    print(judge_type)
    df_selected = df_long[df_long['judge_type'] == judge_type]

    # 根据 model_name 和 jailbreak_type 聚合 ASR 值，使用平均值
    df_pivot = df_selected.pivot_table(index=['model_name'], columns='jailbreak_type', values='ASR', aggfunc='mean') / 10 #.fillna(0)

    # 按照总和从小到大排序（可以选择使用其他排序方式）
    df_pivot = df_pivot.loc[(-df_pivot.sum(axis=1)).sort_values().index]

    # 绘制堆积柱状图
    plt.figure(figsize=(12, 8))
    ax = df_pivot.plot(kind='barh', stacked=True, figsize=(12, 8), colormap='tab10', width=0.67)

    # 在柱子上添加文本（ASR百分比）
    group = len(ax.patches) // 10
    for i in range(group):
        total_width = 0
        for j in range(10):
            total_width += ax.patches[j * group + i].get_width()

        x_position = total_width + .1 # ax.patches[9 * group + i].get_x() + ax.patches[9 * group + i].get_width() + 0.01
        y_position = ax.patches[9 * group + i].get_y() + ax.patches[9 * group + i].get_height() / 2

        ax.text(
            x_position,  # 设置文本的水平位置，稍微偏离柱子的右侧
            y_position,  # 设置文本的垂直位置，居中于柱子
            f'{total_width:.1f}%',  # 显示 ASR 值，保留一位小数
            ha='left',  # 水平对齐方式
            va='center',  # 垂直对齐方式
            fontsize=11,  # 字体大小
            color='black'  # 文字颜色
        )

    # 设置标题和标签
    plt.title(f"攻击成功率 (ASR) (%)", fontsize=16)
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper right', fontsize=11, frameon=False)
    sns.despine()

    # 保存文件
    os.makedirs("figures_new/overview", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"figures_new/overview/stacked_ASR_{judge_type}.pdf", bbox_inches='tight', pad_inches=0.1)

    # 显示图形
    plt.show()
    plt.close()
#%%
df_long
#%% md
### 2. Jailbreak Type Analysis
#%%
numeric_columns = df.select_dtypes(include=["number"]).columns
df_avg = df.groupby(["model_name", "jailbreak_type"])[numeric_columns].mean()
df_avg.reset_index(inplace=True)
#%%
df_avg
#%%
for metric in judge_cls:
    for jailbreak_type in categories:
        df_sorted = df_avg[df_avg["jailbreak_type"] == jailbreak_type].sort_values(by=metric, ascending=True)

        plt.figure(figsize=(12, 8))
        ax = sns.barplot(data=df_sorted, y="model_name", x=metric, palette="Blues_d", hue="model_name")

        for p in ax.patches:
            width = p.get_width()
            ax.text(
                width + 0.01,
                p.get_y() + p.get_height() / 2,
                f'{width:.1f}%',
                ha='left',
                va='center',
                fontsize=11,
                color='black'
            )

        plt.title(f"ASR under the ${jailbreak_type}$ Attack (%)", fontsize=16)
        plt.xlabel('')
        plt.ylabel('')
        sns.despine()

        safe_jailbreak_type = jailbreak_type.replace("/", "_").replace(" ", "-")
        os.makedirs(f"figures_new/jailbreak_type/barplot/{safe_jailbreak_type}", exist_ok=True)
        print(metric)
        plt.savefig(f"figures_new/jailbreak_type/barplot/{safe_jailbreak_type}/ASR_{metric}.pdf", bbox_inches='tight', pad_inches=0.1)

        plt.show()
        plt.close()
#%%
# 循环每个模型，并为每个模型绘制饼状图
for metric in judge_cls:
# metric = "PAIR_gpt-4o-2024-11-20"
    for model_name in df_avg["model_name"].unique():
        # 获取每个模型的对应数据
        df_selected = df_avg[df_avg["model_name"] == model_name]

        # 获取每个模型在不同越狱类别下的 ASR（该 metric 下）
        asr_values = df_selected[["jailbreak_type", metric]].sort_values(by=metric, ascending=True)

        # 处理成饼图的数据格式
        labels = ["Safe", *asr_values['jailbreak_type']]
        sizes = asr_values[metric] / 10  # 除以10 转换为百分比
        sizes = [100 - sum(sizes)] + list(sizes)  # 添加 Safe 的 ASR
        explode = [0.13] * len(labels)  # 使每一块都稍微突出一点
        explode[0] = 0.

        # 绘制饼状图
        plt.figure(figsize=(8, 8))
        plt.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            explode=explode,
            colors=sns.color_palette("tab20", len(labels)),
            pctdistance=0.95,
            labeldistance=1.1
        )

        # 设置标题
        plt.title(f"ASR of {model_name})", fontsize=14)

        # 显示图形
        plt.axis('equal')  # 保证饼图是圆形的
        plt.tight_layout()

        # 保存或显示图形
        safe_model_name = model_name.replace("/", "_").replace(" ", "-")
        os.makedirs(f"figures_new/pie_model/{metric}", exist_ok=True)
        plt.savefig(f"figures_new/pie_model/{metric}/ASR_{safe_model_name}.pdf", bbox_inches='tight', pad_inches=0.1)

        # plt.show()
        plt.close()
#%%
numeric_columns = df.select_dtypes(include=["number"]).columns
df_avg = df.groupby(["jailbreak_type"])[numeric_columns].mean()
df_avg.reset_index(inplace=True)

df_avg
#%%
for metric in judge_cls:

    # 获取每个模型在不同越狱类别下的 ASR（该 metric 下）
    asr_values = df_avg[["jailbreak_type", metric]].sort_values(by=metric, ascending=True)

    # 处理成饼图的数据格式
    labels = ["Safe", *asr_values['jailbreak_type']]
    sizes = asr_values[metric] / 10  # 除以10 转换为百分比
    sizes = [100 - sum(sizes)] + list(sizes)  # 添加 Safe 的 ASR
    explode = [0.13] * len(labels)  # 使每一块都稍微突出一点
    explode[0] = 0.

    # 绘制饼状图
    plt.figure(figsize=(12, 12))
    plt.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=0,
        explode=explode,
        colors=sns.color_palette("tab20", len(labels)),
        pctdistance=0.95,
        labeldistance=1.05
    )

    # 设置标题
    plt.title(f"Average ASR of different LLMs (%)", fontsize=14)

    # 显示图形
    plt.axis('equal')  # 保证饼图是圆形的
    plt.tight_layout()


    print(metric)
    plt.savefig(f"figures_new/pie_model/={metric}.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()
#%% md
### Heatmap Analysis
#%%
numeric_columns = df.select_dtypes(include=["number"]).columns
df_avg = df.groupby(["model_name", "jailbreak_type"])[numeric_columns].mean()
df_avg.reset_index(inplace=True)
#%%
df_avg
#%%
for metric in judge_cls:
    df_selected = df_avg[["model_name", "jailbreak_type", metric]]

    df_pivot = df_avg.pivot_table(index='jailbreak_type', columns='model_name', values=metric, aggfunc='mean')

    # Add Total row  &  Sorted
    df_pivot.loc["$Average$"] = df_pivot.sum() / 10
    df_pivot = df_pivot[df_pivot.loc["$Average$"].sort_values(ascending=True).index]

    plt.figure(figsize=(19, 4.5))
    sns.heatmap(
        data=df_pivot,
        cmap="Blues",
        annot=True,
        fmt="2.0f",
        linewidths=0.5,
        cbar_kws={'label': 'ASR (%)'}
    )

    plt.title(f"Average ASR of different LLMs (%)", fontsize=14)
    plt.xlabel("Model")
    plt.ylabel("Jailbreak Type")
    plt.xticks(rotation=45, ha='right')

    safe_metric = metric.replace("/", "_").replace(" ", "-")
    os.makedirs(f"figures_new/heatmap/model-jailbreak_type", exist_ok=True)
    print(metric)
    plt.savefig(f"figures_new/heatmap/model-jailbreak_type/{safe_metric}.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()
#%%
numeric_columns = df.select_dtypes(include=["number"]).columns
df_avg = df.groupby(["model_name", "attack_method"])[numeric_columns].mean()
df_avg.reset_index(inplace=True)

for metric in judge_cls:
    df_selected = df_avg[["model_name", "attack_method", metric]]

    df_pivot = df_avg.pivot_table(index='attack_method', columns='model_name', values=metric, aggfunc='mean')

    # Replace NaN by Zero
    df_pivot = df_pivot.fillna(0)

    # Add Total row  &  Sorted
    df_pivot.loc["$Average$"] = df_pivot.sum() / 24
    df_pivot = df_pivot[df_pivot.loc["$Average$"].sort_values(ascending=True).index]

    plt.figure(figsize=(16, 6))
    sns.heatmap(
        data=df_pivot,
        cmap="Blues",
        annot=True,
        fmt="2.0f",
        linewidths=0.5,
        cbar_kws={'label': 'ASR (%)'}
    )

    plt.title(f"Average ASR of different LLMs (%)", fontsize=14)
    plt.xlabel("Model")
    plt.ylabel("Attack Method")
    plt.xticks(rotation=45, ha='right')

    safe_metric = metric.replace("/", "_").replace(" ", "-")
    os.makedirs(f"figures_new/heatmap/model-attack_method", exist_ok=True)
    print(metric)
    plt.savefig(f"figures_new/heatmap/model-attack_method/{safe_metric}.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()
#%%
numeric_columns = df.select_dtypes(include=["number"]).columns
df_avg = df.groupby(["jailbreak_type", "attack_method"])[numeric_columns].mean()
df_avg.reset_index(inplace=True)

for metric in judge_cls:
    df_selected = df_avg[["attack_method", "jailbreak_type", metric]]

    df_pivot = df_avg.pivot_table(index='jailbreak_type', columns='attack_method', values=metric, aggfunc='mean')

    # Replace NaN by Zero
    df_pivot = df_pivot.fillna(0)

    # Add Total row  &  Sorted
    df_pivot.loc["$Average$"] = df_pivot.sum() / 10
    df_pivot = df_pivot[df_pivot.loc["$Average$"].sort_values(ascending=True).index]

    plt.figure(figsize=(10, 4.5))
    sns.heatmap(
        data=df_pivot,
        cmap="Blues",
        annot=True,
        fmt="2.0f",
        linewidths=0.5,
        cbar_kws={'label': 'ASR (%)'}
    )

    plt.title(f"Average ASR of different LLMs (%)", fontsize=14)
    plt.ylabel("Jailbreak Type")
    plt.xlabel("Attack Method")
    plt.xticks(rotation=45, ha='right')

    safe_metric = metric.replace("/", "_").replace(" ", "-")
    os.makedirs(f"figures_new/heatmap/jailbreak_type-attack_method", exist_ok=True)
    print(metric)
    plt.savefig(f"figures_new/heatmap/jailbreak_type-attack_method/{safe_metric}.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()
#%%
for metric in judge_cls:
    for jailbreak_type in df["jailbreak_type"].unique():
        df_selected = df[["model_name", "attack_method", metric]]
        df_selected = df_selected[df["jailbreak_type"] == jailbreak_type]

        df_pivot = df_selected.pivot_table(index='attack_method', columns='model_name', values=metric, aggfunc='mean')
        df_pivot = df_pivot.fillna(0)

        # Add Total row  &  Sorted
        df_pivot.loc["$Average$"] = df_pivot.sum() / 24
        df_pivot = df_pivot[df_pivot.loc["$Average$"].sort_values(ascending=True).index]

        plt.figure(figsize=(17, 7))
        sns.heatmap(
            data=df_pivot,
            cmap="Blues",
            annot=True,
            fmt="2.0f",
            linewidths=0.5,
            cbar_kws={'label': 'ASR (%)'}
        )

        plt.title(f"Average ASR under ${jailbreak_type}$ Attack (%)", fontsize=14)
        plt.xlabel("Model")
        plt.ylabel("Jailbreak Type")
        plt.xticks(rotation=45, ha='right')

        safe_metric = metric.replace("/", "_").replace(" ", "-")
        os.makedirs(f"figures_new/jailbreak_type/heatmap/{jailbreak_type}", exist_ok=True)
        print(metric)
        plt.savefig(f"figures_new/jailbreak_type/heatmap/{jailbreak_type}/{safe_metric}.pdf", bbox_inches='tight', pad_inches=0.1)
        plt.show()
        plt.close()
#%%

#%%

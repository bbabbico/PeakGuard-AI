# -*- coding: utf-8 -*-
import os, sys, warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

df = pd.read_csv('output_v2/final_camera_reallocation_targets.csv')

# 각 행의 사고유형을 분리하여 리스트로 만듦
records = []
for idx, row in df.iterrows():
    if pd.isna(row['사고분석유형명']): continue
    types = [t.strip() for t in row['사고분석유형명'].split('/')]
    for t in types:
        records.append({'우선순위': row['우선순위'], '사고유형': t})

df_types = pd.DataFrame(records)

# 우선순위별 사고유형 비율 계산
type_counts = df_types.groupby(['우선순위', '사고유형']).size().unstack(fill_value=0)
type_ratios = type_counts.div(type_counts.sum(axis=1), axis=0) * 100

# 카메라 단속 가능 여부 매핑
camera_preventable = ['신호위반', '안전거리미확보', '과속', '중앙선침범']
unpreventable = ['기타', 'U턴중', '보행자보호위반']

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('카메라 재배치 실효성 검증: 타겟 구역별 사고 예방 가능성 비교', fontsize=22, fontweight='bold', y=0.96)

import matplotlib.patches as mpatches
prev_patch = mpatches.Patch(color='#3498db', label='카메라로 예방/단속 가능')
unprev_patch = mpatches.Patch(color='#95a5a6', label='카메라로 예방 어려움')

priorities = ['1순위 (사망/사고다발)', '2순위 (일반부족 상위10%)', '3순위 (일반부족 나머지)']
titles = ['[1순위 타겟] 구역 상세 사고 유형', '[2순위 타겟] 구역 상세 사고 유형', '[3순위 후순위] 구역 상세 사고 유형']

for i, (ax, p_name, title) in enumerate(zip([axes[0,0], axes[0,1], axes[1,0]], priorities, titles)):
    p_data = type_ratios.loc[p_name].sort_values(ascending=True)
    colors = ['#3498db' if x in camera_preventable else '#95a5a6' for x in p_data.index]
    
    ax.barh(p_data.index, p_data.values, color=colors, edgecolor='white')
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel('사고 발생 비율 (%)')
    ax.set_xlim(0, 55) # 통일된 x축으로 크기 비교 용이
    for j, v in enumerate(p_data.values):
        ax.text(v + 1.0, j, f"{v:.1f}%", va='center', fontweight='bold', fontsize=11)
    if i == 0:
        ax.legend(handles=[prev_patch, unprev_patch], loc='lower right', fontsize=11)

# 4번째 subplot: 카메라 예방 가능 비율 종합 비교
ax4 = axes[1, 1]
prev_ratios = []
for p in priorities:
    ratio = type_ratios.loc[p, type_ratios.columns.intersection(camera_preventable)].sum()
    prev_ratios.append(ratio)

x = np.arange(len(priorities))
bars = ax4.bar(x, prev_ratios, color='#e74c3c', width=0.5, edgecolor='white')
ax4.set_xticks(x)
labels = [label.replace(" ", "\n") for label in priorities]
ax4.set_xticklabels(labels, fontsize=13, fontweight='bold')
ax4.set_title('우선순위 그룹별 [카메라 예방 가능 사고] 총합 비교', fontsize=16, fontweight='bold')
ax4.set_ylabel('카메라 단속 가능 사고 비율 (%)')
ax4.set_ylim(0, 55)

for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('output_v2/09_priority_effectiveness.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 09_priority_effectiveness.png")

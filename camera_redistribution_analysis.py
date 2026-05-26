# -*- coding: utf-8 -*-
"""
=============================================================================
  단속 카메라 재배치 근거 분석
  - 사고위험지역 데이터 × 무인교통단속카메라 데이터 EDA·군집화
=============================================================================
출력: output/ 폴더에 PNG 차트 + CSV 결과
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pyproj import Transformer
from scipy.spatial import cKDTree

# ── 한글 폰트 설정 ─────────────────────────────────────────────
mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# ── 출력 폴더 ──────────────────────────────────────────────────
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("  단속 카메라 재배치 근거 분석 시작")
print("=" * 70)


# ╔═══════════════════════════════════════════════════════════════╗
# ║  PHASE 1: 데이터 로드 & 전처리                                  ║
# ╚═══════════════════════════════════════════════════════════════╝
print("\n[Phase 1] 데이터 로드 및 전처리")

# 1-1. 사고위험지역
df_risk = pd.read_csv('RiskArea.csv', encoding='cp949')
df_risk.rename(columns={'츙사고건수': '총사고건수'}, inplace=True)   # 오타 수정

# 1-2. 무인교통단속카메라
df_cam = pd.read_csv('경찰청_무인교통단속카메라_20260406.csv', encoding='utf-8')

print(f"  사고위험지역: {df_risk.shape[0]:,}건,  카메라: {df_cam.shape[0]:,}건")

# 1-3. 카메라 위경도 → UTM-K 변환
transformer = Transformer.from_crs("EPSG:4326", "EPSG:5178", always_xy=True)
cam_x, cam_y = transformer.transform(
    df_cam['경도'].values, df_cam['위도'].values
)
df_cam['utmk_x'] = cam_x
df_cam['utmk_y'] = cam_y

# 1-4. 단속구분 라벨 매핑
단속구분_map = {1: '과속', 2: '신호위반', 3: '과속+신호', 99: '기타'}
df_cam['단속구분명'] = df_cam['단속구분'].map(단속구분_map).fillna('기타')

# 1-5. 보호구역 라벨 매핑
보호구역_map = {1.0: '어린이보호구역', 2.0: '스쿨존외보호구역', 99.0: '비보호구역'}
df_cam['보호구역명'] = df_cam['보호구역구분'].map(보호구역_map).fillna('미분류')

# 1-6. 사고위험지역에 시도/시군구 매핑을 위해 시군구코드 앞 2자리 → 시도
시도코드_map = {
    11: '서울', 26: '부산', 27: '대구', 28: '인천',
    29: '광주', 30: '대전', 31: '울산', 36: '세종',
    41: '경기', 42: '강원', 43: '충북', 44: '충남',
    45: '전북', 46: '전남', 47: '경북', 48: '경남', 50: '제주'
}
df_risk['시도코드'] = df_risk['시군구코드'] // 1000
df_risk['시도명_short'] = df_risk['시도코드'].map(시도코드_map)

# 카메라 시도명을 축약
시도명_축약 = {
    '서울특별시': '서울', '부산광역시': '부산', '대구광역시': '대구',
    '인천광역시': '인천', '광주광역시': '광주', '대전광역시': '대전',
    '울산광역시': '울산', '세종특별자치시': '세종', '경기도': '경기',
    '강원특별자치도': '강원', '충청북도': '충북', '충청남도': '충남',
    '전북특별자치도': '전북', '전라남도': '전남',
    '경상북도': '경북', '경상남도': '경남', '제주특별자치도': '제주'
}
df_cam['시도명_short'] = df_cam['시도명'].map(시도명_축약)

print("  좌표 변환 및 전처리 완료")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  PHASE 2: 개별 EDA 시각화                                      ║
# ╚═══════════════════════════════════════════════════════════════╝
print("\n[Phase 2] EDA 시각화")

# ── 2-1. 카메라 EDA ────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('무인교통단속카메라 EDA', fontsize=20, fontweight='bold', y=0.98)

# (a) 시도별 카메라 수
cam_sido = df_cam['시도명_short'].value_counts().sort_values(ascending=True)
colors_cam = plt.cm.Blues(np.linspace(0.3, 0.9, len(cam_sido)))
ax = axes[0, 0]
bars = ax.barh(cam_sido.index, cam_sido.values, color=colors_cam, edgecolor='white')
ax.set_title('시도별 단속 카메라 설치 수', fontsize=14, fontweight='bold')
ax.set_xlabel('카메라 수')
for bar, val in zip(bars, cam_sido.values):
    ax.text(val + 30, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', fontsize=9)

# (b) 단속구분별 비율
ax = axes[0, 1]
단속_counts = df_cam['단속구분명'].value_counts()
colors_pie = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
wedges, texts, autotexts = ax.pie(
    단속_counts.values, labels=단속_counts.index,
    autopct='%1.1f%%', colors=colors_pie,
    startangle=90, textprops={'fontsize': 12}
)
ax.set_title('단속구분별 카메라 비율', fontsize=14, fontweight='bold')

# (c) 제한속도별 분포
ax = axes[1, 0]
speed_order = sorted(df_cam['제한속도'].unique())
speed_counts = df_cam['제한속도'].value_counts().reindex(speed_order)
colors_speed = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(speed_order)))
ax.bar(range(len(speed_order)), speed_counts.values, color=colors_speed, edgecolor='white')
ax.set_xticks(range(len(speed_order)))
ax.set_xticklabels([f'{s}km/h' for s in speed_order], rotation=45)
ax.set_title('제한속도별 카메라 수', fontsize=14, fontweight='bold')
ax.set_ylabel('카메라 수')
for i, v in enumerate(speed_counts.values):
    ax.text(i, v + 50, f'{v:,}', ha='center', fontsize=9)

# (d) 보호구역 구분 분포
ax = axes[1, 1]
보호_counts = df_cam['보호구역명'].value_counts()
colors_zone = ['#1abc9c', '#e67e22', '#9b59b6', '#95a5a6']
ax.bar(보호_counts.index, 보호_counts.values,
       color=colors_zone[:len(보호_counts)], edgecolor='white')
ax.set_title('보호구역 구분별 카메라 수', fontsize=14, fontweight='bold')
ax.set_ylabel('카메라 수')
for i, v in enumerate(보호_counts.values):
    ax.text(i, v + 100, f'{v:,}', ha='center', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '01_eda_camera.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → 01_eda_camera.png 저장")

# ── 2-2. 사고위험지역 EDA ──────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('사고위험지역 EDA', fontsize=20, fontweight='bold', y=0.98)

# (a) 연도별 사고 건수 합계 추이
year_stats = df_risk.groupby('연도코드').agg(
    사고건수합=('총사고건수', 'sum'),
    사망자합=('총사망자수', 'sum'),
    중상자합=('총중상자수', 'sum')
).reset_index()
ax = axes[0, 0]
ax.plot(year_stats['연도코드'], year_stats['사고건수합'], 'o-',
        color='#e74c3c', linewidth=2.5, markersize=8, label='총사고건수')
ax.fill_between(year_stats['연도코드'], year_stats['사고건수합'], alpha=0.15, color='#e74c3c')
ax.set_title('연도별 사고위험지역 총 사고건수 추이', fontsize=14, fontweight='bold')
ax.set_xlabel('연도')
ax.set_ylabel('사고 건수')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# (b) 시도별 사고위험지역 수 & 평균 사고건수
risk_sido = df_risk.groupby('시도명_short').agg(
    위험지역수=('사고위험지역id', 'count'),
    평균사고건수=('총사고건수', 'mean')
).sort_values('위험지역수', ascending=True)
ax = axes[0, 1]
colors_risk = plt.cm.Reds(np.linspace(0.3, 0.9, len(risk_sido)))
ax.barh(risk_sido.index, risk_sido['위험지역수'], color=colors_risk, edgecolor='white')
ax.set_title('시도별 사고위험지역 수', fontsize=14, fontweight='bold')
ax.set_xlabel('위험지역 수')
for i, (idx, row) in enumerate(risk_sido.iterrows()):
    ax.text(row['위험지역수'] + 5, i, f"{int(row['위험지역수']):,}", va='center', fontsize=9)

# (c) 사고 유형 TOP 10
ax = axes[1, 0]
# 개별 사고유형 추출 (슬래시로 분리)
all_types = []
for t in df_risk['사고분석유형명'].dropna():
    for sub in t.split(' / '):
        all_types.append(sub.strip())
type_series = pd.Series(all_types)
type_top10 = type_series.value_counts().head(10)
colors_type = plt.cm.Set2(np.linspace(0, 1, 10))
ax.barh(type_top10.index[::-1], type_top10.values[::-1],
        color=colors_type, edgecolor='white')
ax.set_title('사고 유형 TOP 10 (개별 유형 분리)', fontsize=14, fontweight='bold')
ax.set_xlabel('빈도')

# (d) 사고 심각도 분포 (박스플롯)
ax = axes[1, 1]
severity_cols = ['총사고건수', '총사망자수', '총중상자수', '총경상자수']
df_severity = df_risk[severity_cols].melt(var_name='지표', value_name='건수')
sns.boxplot(data=df_severity, x='지표', y='건수', ax=ax,
            palette='Set2', showfliers=False)
ax.set_title('사고 심각도 지표 분포', fontsize=14, fontweight='bold')
ax.set_ylabel('건수')
ax.set_xticklabels(ax.get_xticklabels(), rotation=15)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '02_eda_accident.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → 02_eda_accident.png 저장")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  PHASE 3: 공간 매칭 & 연관성 분석                                ║
# ╚═══════════════════════════════════════════════════════════════╝
print("\n[Phase 3] 공간 매칭 및 연관성 분석")

# ── 3-1. 시도 단위 매칭 ────────────────────────────────────────
# 시도별 카메라 수
cam_by_sido = df_cam.groupby('시도명_short').size().reset_index(name='카메라수')

# 시도별 사고 합계 (최신 연도 2024 기준)
risk_latest = df_risk[df_risk['연도코드'] == df_risk['연도코드'].max()]
risk_by_sido = risk_latest.groupby('시도명_short').agg(
    위험지역수=('사고위험지역id', 'count'),
    총사고건수합=('총사고건수', 'sum'),
    총사망자수합=('총사망자수', 'sum'),
    총중상자수합=('총중상자수', 'sum'),
    평균사고건수=('총사고건수', 'mean')
).reset_index()

# 병합
merged_sido = pd.merge(cam_by_sido, risk_by_sido, on='시도명_short', how='inner')
merged_sido['카메라_대비_사고'] = merged_sido['총사고건수합'] / merged_sido['카메라수']
merged_sido['카메라_대비_사망'] = merged_sido['총사망자수합'] / merged_sido['카메라수']

print(f"  시도 매칭: {len(merged_sido)}개 시도")

# ── 3-2. 사고위험지역 반경 내 카메라 존재 분석 ─────────────────
RADIUS = 500  # 미터

# KD-Tree로 사고위험지역 중심점 근처 카메라 탐색
risk_coords = df_risk[['중심점utmkx좌표', '중심점utmky좌표']].dropna().values
cam_coords = df_cam[['utmk_x', 'utmk_y']].values

tree_cam = cKDTree(cam_coords)

# 각 사고위험지역 반경 내 카메라 수 계산
cam_in_radius = []
for rx, ry in risk_coords:
    indices = tree_cam.query_ball_point([rx, ry], r=RADIUS)
    cam_in_radius.append(len(indices))

df_risk_valid = df_risk.dropna(subset=['중심점utmkx좌표', '중심점utmky좌표']).copy()
df_risk_valid['반경내카메라수'] = cam_in_radius
df_risk_valid['카메라유무'] = (df_risk_valid['반경내카메라수'] > 0).astype(int)

has_cam = df_risk_valid[df_risk_valid['카메라유무'] == 1]
no_cam = df_risk_valid[df_risk_valid['카메라유무'] == 0]

print(f"  사고위험지역 {len(df_risk_valid):,}건 중 반경 {RADIUS}m 내 카메라 있음: "
      f"{len(has_cam):,}건 ({100*len(has_cam)/len(df_risk_valid):.1f}%)")
print(f"  카메라 없는 사고위험지역: {len(no_cam):,}건 ({100*len(no_cam)/len(df_risk_valid):.1f}%)")

# ── 3-3. 연관성 시각화 ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('카메라-사고 연관성 분석', fontsize=20, fontweight='bold', y=0.98)

# (a) 시도별 카메라 수 vs 사고건수 산점도
ax = axes[0, 0]
ax.scatter(merged_sido['카메라수'], merged_sido['총사고건수합'],
           s=merged_sido['총사망자수합'] * 30 + 50,
           c='#e74c3c', alpha=0.7, edgecolors='white', linewidth=1.5)
for _, row in merged_sido.iterrows():
    ax.annotate(row['시도명_short'],
                (row['카메라수'], row['총사고건수합']),
                fontsize=9, ha='center', va='bottom',
                xytext=(0, 8), textcoords='offset points')
corr_val = merged_sido['카메라수'].corr(merged_sido['총사고건수합'])
ax.set_title(f'시도별 카메라 수 vs 사고 건수 (r={corr_val:.3f})', fontsize=14, fontweight='bold')
ax.set_xlabel('카메라 수')
ax.set_ylabel('총 사고 건수')
ax.grid(alpha=0.3)
# 추세선
z = np.polyfit(merged_sido['카메라수'], merged_sido['총사고건수합'], 1)
p = np.poly1d(z)
x_line = np.linspace(merged_sido['카메라수'].min(), merged_sido['카메라수'].max(), 100)
ax.plot(x_line, p(x_line), '--', color='#3498db', linewidth=2, alpha=0.7, label='추세선')
ax.legend()

# (b) 카메라 유무에 따른 평균 사고건수 비교
ax = axes[0, 1]
cam_comparison = df_risk_valid.groupby('카메라유무').agg(
    평균사고건수=('총사고건수', 'mean'),
    평균사망자수=('총사망자수', 'mean'),
    평균중상자수=('총중상자수', 'mean')
).reset_index()
cam_comparison['카메라유무_label'] = cam_comparison['카메라유무'].map({0: '카메라 없음', 1: '카메라 있음'})

x_pos = np.arange(3)
width = 0.35
bars1 = ax.bar(x_pos - width/2,
               cam_comparison[cam_comparison['카메라유무'] == 0][['평균사고건수', '평균사망자수', '평균중상자수']].values[0],
               width, label='카메라 없음', color='#e74c3c', alpha=0.85)
bars2 = ax.bar(x_pos + width/2,
               cam_comparison[cam_comparison['카메라유무'] == 1][['평균사고건수', '평균사망자수', '평균중상자수']].values[0],
               width, label='카메라 있음', color='#3498db', alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels(['평균 사고건수', '평균 사망자수', '평균 중상자수'])
ax.set_title('카메라 유무별 사고 지표 비교 (반경 500m)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                f'{h:.2f}', ha='center', va='bottom', fontsize=9)

# (c) 시도별 카메라 대비 사고 비율
ax = axes[1, 0]
ratio_sorted = merged_sido.sort_values('카메라_대비_사고', ascending=True)
colors_ratio = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(ratio_sorted)))
ax.barh(ratio_sorted['시도명_short'], ratio_sorted['카메라_대비_사고'],
        color=colors_ratio, edgecolor='white')
ax.set_title('시도별 카메라 1대당 사고 건수', fontsize=14, fontweight='bold')
ax.set_xlabel('카메라 1대당 사고 건수')
ax.axvline(x=ratio_sorted['카메라_대비_사고'].mean(), color='#e74c3c',
           linestyle='--', linewidth=2, label=f"평균: {ratio_sorted['카메라_대비_사고'].mean():.2f}")
ax.legend()

# (d) 사고위험지역 반경 내 카메라 수 분포
ax = axes[1, 1]
cam_dist = df_risk_valid['반경내카메라수'].value_counts().sort_index().head(15)
ax.bar(cam_dist.index, cam_dist.values, color='#2ecc71', edgecolor='white', alpha=0.85)
ax.set_title('사고위험지역 반경 500m 내 카메라 수 분포', fontsize=14, fontweight='bold')
ax.set_xlabel('카메라 수')
ax.set_ylabel('사고위험지역 수')
for i, v in zip(cam_dist.index, cam_dist.values):
    ax.text(i, v + 20, f'{v:,}', ha='center', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '03_correlation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → 03_correlation.png 저장")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  PHASE 4: K-Means 군집화                                      ║
# ╚═══════════════════════════════════════════════════════════════╝
print("\n[Phase 4] K-Means 군집화")

# ── 4-1. 시군구 단위 특성 벡터 구성 ────────────────────────────
# 카메라: 시군구별 수
cam_by_sigungu = df_cam.groupby(['시도명_short', '시군구명']).agg(
    카메라수=('무인교통단속카메라관리번호', 'count'),
    평균제한속도=('제한속도', 'mean')
).reset_index()

# 사고: 시군구코드 → 시도 + 최신연도 기준
risk_by_sigungu = risk_latest.groupby(['시도명_short', '시군구코드']).agg(
    위험지역수=('사고위험지역id', 'count'),
    총사고건수합=('총사고건수', 'sum'),
    총사망자수합=('총사망자수', 'sum'),
    총중상자수합=('총중상자수', 'sum'),
    평균사고건수=('총사고건수', 'mean')
).reset_index()

# ── 시도 단위 군집화 (효율성 기반) ──────────────────────────────
# 핵심: 절대값이 아닌 '효율성 지표'를 기반으로 군집화
# → 카메라 1대당 사고 비율, 위험지역당 카메라 수, 사고 심각도(사망률)
merged_sido['위험지역당카메라'] = merged_sido['카메라수'] / merged_sido['위험지역수']
merged_sido['사망률'] = merged_sido['총사망자수합'] / merged_sido['총사고건수합'].replace(0, 1)
merged_sido['중상률'] = merged_sido['총중상자수합'] / merged_sido['총사고건수합'].replace(0, 1)

features = merged_sido[['카메라_대비_사고', '위험지역당카메라', '사망률', '중상률']].copy()
feature_names = ['카메라_대비_사고', '위험지역당카메라', '사망률', '중상률']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ── 4-2. Elbow Method ─────────────────────────────────────────
inertias = []
K_range = range(2, min(len(merged_sido), 10))
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(features_scaled)
    inertias.append(km.inertia_)

# ── 4-3. 최적 k로 클러스터링 (k=4로 설정: 과잉/적정/부족/위험) ─
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
merged_sido['cluster'] = kmeans.fit_predict(features_scaled)

# ── 4-4. 군집 해석 ─────────────────────────────────────────────
cluster_stats = merged_sido.groupby('cluster').agg(
    평균카메라수=('카메라수', 'mean'),
    평균사고건수=('총사고건수합', 'mean'),
    평균사망자수=('총사망자수합', 'mean'),
    카메라대비사고=('카메라_대비_사고', 'mean'),
    위험지역당카메라_평균=('위험지역당카메라', 'mean')
).reset_index()

# 군집 라벨 부여: 효율성 순위 기반 분류
# 효율성 점수 = 위험지역당카메라(공급) / 카메라대비사고(수요)
# → 높을수록 과잉(카메라 공급 대비 사고 부담 적음)
# → 낮을수록 부족(카메라 공급 대비 사고 부담 큼)
cluster_stats['효율성점수'] = (
    cluster_stats['위험지역당카메라_평균'] / cluster_stats['카메라대비사고'].replace(0, 0.001)
)
cluster_stats = cluster_stats.sort_values('효율성점수', ascending=False)

# 효율성 순위로 4단계 라벨 부여 (1위=과잉, 2위=적정, 3위=부족, 4위=위험)
label_order = [
    '과잉 구역 (카메라↑ 사고↓)',
    '적정 구역 (카메라↓ 사고↓)',
    '부족 구역 (카메라↓ 사고↑)',
    '위험 구역 (카메라↑ 사고↑)'
]
label_map = {}
for i, (_, row) in enumerate(cluster_stats.iterrows()):
    label_map[row['cluster']] = label_order[i]

merged_sido['군집라벨'] = merged_sido['cluster'].map(label_map)

print("  군집 결과:")
for label in sorted(label_map.values()):
    subset = merged_sido[merged_sido['군집라벨'] == label]
    regions = ', '.join(subset['시도명_short'].tolist())
    print(f"    {label}: {regions}")

# ── 4-5. 군집화 시각화 ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('K-Means 군집화 분석 결과', fontsize=20, fontweight='bold', y=0.98)

cluster_colors = {
    '과잉 구역 (카메라↑ 사고↓)': '#3498db',
    '적정 구역 (카메라↓ 사고↓)': '#2ecc71',
    '부족 구역 (카메라↓ 사고↑)': '#e74c3c',
    '위험 구역 (카메라↑ 사고↑)': '#f39c12'
}

# (a) Elbow Chart
ax = axes[0, 0]
ax.plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2,
           label=f'선택: k={optimal_k}')
ax.set_title('Elbow Method (최적 k 결정)', fontsize=14, fontweight='bold')
ax.set_xlabel('클러스터 수 (k)')
ax.set_ylabel('Inertia')
ax.legend()
ax.grid(alpha=0.3)

# (b) 카메라 수 vs 사고건수 군집 산점도
ax = axes[0, 1]
for label, color in cluster_colors.items():
    subset = merged_sido[merged_sido['군집라벨'] == label]
    ax.scatter(subset['카메라수'], subset['총사고건수합'],
               s=150, c=color, label=label, edgecolors='white',
               linewidth=1.5, alpha=0.85, zorder=5)
    for _, row in subset.iterrows():
        ax.annotate(row['시도명_short'],
                    (row['카메라수'], row['총사고건수합']),
                    fontsize=9, ha='center', va='bottom',
                    xytext=(0, 10), textcoords='offset points')
ax.set_title('시도별 군집 분류 (카메라 수 vs 사고 건수)', fontsize=14, fontweight='bold')
ax.set_xlabel('카메라 수')
ax.set_ylabel('총 사고 건수')
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.3)

# (c) 군집별 평균 지표 비교 (그룹 바 차트)
ax = axes[1, 0]
cluster_means = merged_sido.groupby('군집라벨')[['카메라_대비_사고', '위험지역당카메라',
                                               '사망률', '중상률']].mean()
# 정규화
cluster_means_norm = cluster_means.copy()
for col in cluster_means.columns:
    max_val = cluster_means[col].max()
    if max_val > 0:
        cluster_means_norm[col] = cluster_means[col] / max_val

x_pos = np.arange(len(cluster_means_norm.columns))
width = 0.2
for i, (label, row) in enumerate(cluster_means_norm.iterrows()):
    short_label = label.split('(')[0].strip()
    color = cluster_colors.get(label, '#95a5a6')
    ax.bar(x_pos + i * width, row.values, width,
           label=short_label, color=color, alpha=0.85)
ax.set_xticks(x_pos + width * 1.5)
ax.set_xticklabels(['1대당사고', '지역당카메라', '사망률', '중상률'], fontsize=10)
ax.set_title('군집별 평균 지표 비교 (정규화)', fontsize=14, fontweight='bold')
ax.set_ylabel('정규화 값')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# (d) 카메라 1대당 사고 건수 by 군집
ax = axes[1, 1]
for label, color in cluster_colors.items():
    subset = merged_sido[merged_sido['군집라벨'] == label]
    short = label.split('(')[0].strip()
    ax.bar(subset['시도명_short'], subset['카메라_대비_사고'],
           color=color, label=short, edgecolor='white', alpha=0.85)
ax.set_title('시도별 카메라 1대당 사고 건수 (군집별 색상)', fontsize=14, fontweight='bold')
ax.set_ylabel('카메라 1대당 사고 건수')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.axhline(y=merged_sido['카메라_대비_사고'].mean(), color='black',
           linestyle='--', linewidth=1.5, alpha=0.5, label='전체 평균')
ax.legend(fontsize=8, loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '04_clustering.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → 04_clustering.png 저장")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  PHASE 5: 재배치 제안 도출                                      ║
# ╚═══════════════════════════════════════════════════════════════╝
print("\n[Phase 5] 재배치 제안 도출")

# ── 5-1. 카메라 없는 고위험 사고지역 TOP 20 ──────────────────────
no_cam_high_risk = df_risk_valid[df_risk_valid['카메라유무'] == 0].copy()
no_cam_high_risk['위험점수'] = (
    no_cam_high_risk['총사고건수'] * 1.0 +
    no_cam_high_risk['총사망자수'] * 10.0 +
    no_cam_high_risk['총중상자수'] * 5.0
)
no_cam_top = no_cam_high_risk.nlargest(20, '위험점수')[
    ['연도코드', '시도명_short', '사고위험지역명', '총사고건수',
     '총사망자수', '총중상자수', '위험점수', '사고분석유형명']
]

# ── 5-2. 카메라 과잉 지역 (카메라 많은데 사고 적은 위험지역) ────
high_cam_low_risk = df_risk_valid[df_risk_valid['반경내카메라수'] >= 3].copy()
high_cam_low_risk['위험점수'] = (
    high_cam_low_risk['총사고건수'] * 1.0 +
    high_cam_low_risk['총사망자수'] * 10.0 +
    high_cam_low_risk['총중상자수'] * 5.0
)
high_cam_low_risk_sorted = high_cam_low_risk.nsmallest(20, '위험점수')[
    ['연도코드', '시도명_short', '사고위험지역명', '총사고건수',
     '총사망자수', '총중상자수', '반경내카메라수', '위험점수']
]

# ── 5-3. 재배치 종합 시각화 ────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('단속 카메라 재배치 제안', fontsize=20, fontweight='bold', y=0.98)

# (a) 카메라 없는 고위험 지역 TOP 15
ax = axes[0, 0]
top15 = no_cam_high_risk.nlargest(15, '위험점수')
labels = [f"{r['시도명_short']}" for _, r in top15.iterrows()]
ax.barh(range(len(top15)), top15['위험점수'].values,
        color='#e74c3c', alpha=0.85, edgecolor='white')
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_title('카메라 미설치 고위험 지역 TOP 15\n(카메라 신규 설치 필요)', fontsize=13, fontweight='bold')
ax.set_xlabel('위험 점수')
ax.invert_yaxis()

# (b) 카메라 과밀 저위험 지역 TOP 15
ax = axes[0, 1]
top15_over = high_cam_low_risk.nsmallest(15, '위험점수')
labels2 = [f"{r['시도명_short']}({int(r['반경내카메라수'])}대)" for _, r in top15_over.iterrows()]
ax.barh(range(len(top15_over)), top15_over['반경내카메라수'].values,
        color='#3498db', alpha=0.85, edgecolor='white')
ax.set_yticks(range(len(top15_over)))
ax.set_yticklabels(labels2, fontsize=9)
ax.set_title('카메라 과밀 저위험 지역 TOP 15\n(카메라 감축 가능)', fontsize=13, fontweight='bold')
ax.set_xlabel('반경 500m 내 카메라 수')
ax.invert_yaxis()

# (c) 시도별 재배치 방향 화살표 차트
ax = axes[1, 0]
# 과잉 → 부족으로 재배치 방향 표시
merged_plot = merged_sido.sort_values('카메라_대비_사고', ascending=False)
colors_bar = []
for _, row in merged_plot.iterrows():
    if '과잉' in str(row['군집라벨']):
        colors_bar.append('#3498db')
    elif '부족' in str(row['군집라벨']):
        colors_bar.append('#e74c3c')
    elif '위험' in str(row['군집라벨']):
        colors_bar.append('#f39c12')
    else:
        colors_bar.append('#2ecc71')

ax.barh(merged_plot['시도명_short'], merged_plot['카메라_대비_사고'],
        color=colors_bar, edgecolor='white', alpha=0.85)
ax.axvline(x=merged_plot['카메라_대비_사고'].mean(), color='black',
           linestyle='--', linewidth=2, label='전국 평균')
ax.set_title('시도별 카메라 효율성 (1대당 사고 건수)', fontsize=14, fontweight='bold')
ax.set_xlabel('카메라 1대당 사고 건수')
ax.legend()

# 범례
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', label='부족 (카메라 증설 필요)'),
    Patch(facecolor='#f39c12', label='위험 (효과 점검 필요)'),
    Patch(facecolor='#2ecc71', label='적정'),
    Patch(facecolor='#3498db', label='과잉 (카메라 감축 가능)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

# (d) 재배치 요약 텍스트
ax = axes[1, 1]
ax.axis('off')

과잉_list = merged_sido[merged_sido['군집라벨'].str.contains('과잉', na=False)]['시도명_short'].tolist()
부족_list = merged_sido[merged_sido['군집라벨'].str.contains('부족', na=False)]['시도명_short'].tolist()
위험_list = merged_sido[merged_sido['군집라벨'].str.contains('위험', na=False)]['시도명_short'].tolist()
적정_list = merged_sido[merged_sido['군집라벨'].str.contains('적정', na=False)]['시도명_short'].tolist()

no_cam_pct = 100 * len(no_cam) / len(df_risk_valid)

summary_text = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       단속 카메라 재배치 분석 요약
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 분석 데이터
   · 전국 단속 카메라: {len(df_cam):,}대
   · 사고 위험지역: {len(df_risk_valid):,}건

🔍 핵심 발견
   · 사고위험지역 중 반경 500m 내
     카메라가 없는 곳: {no_cam_pct:.1f}%

🔴 카메라 부족 구역 (증설 필요)
   → {', '.join(부족_list) if 부족_list else '해당 없음'}

🟡 위험 구역 (효과 점검 필요)
   → {', '.join(위험_list) if 위험_list else '해당 없음'}

🟢 적정 구역
   → {', '.join(적정_list) if 적정_list else '해당 없음'}

🔵 과잉 구역 (감축 가능)
   → {', '.join(과잉_list) if 과잉_list else '해당 없음'}

💡 제안
   과잉 구역의 카메라 일부를
   부족 구역으로 재배치하여
   전국적 교통 안전 효과를 극대화
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=12, verticalalignment='top',
        fontfamily='Malgun Gothic',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa',
                  edgecolor='#dee2e6', alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, '05_redistribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → 05_redistribution.png 저장")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  결과 CSV 저장                                                 ║
# ╚═══════════════════════════════════════════════════════════════╝
print("\n[결과 저장]")

# 군집 결과
merged_sido.to_csv(os.path.join(OUTPUT_DIR, 'cluster_result.csv'),
                   index=False, encoding='utf-8-sig')
print("  → cluster_result.csv 저장")

# 카메라 없는 고위험지역 TOP 20
no_cam_top.to_csv(os.path.join(OUTPUT_DIR, 'high_risk_no_camera_top20.csv'),
                  index=False, encoding='utf-8-sig')
print("  → high_risk_no_camera_top20.csv 저장")

# 카메라 과잉 저위험지역
high_cam_low_risk_sorted.to_csv(os.path.join(OUTPUT_DIR, 'low_risk_high_camera.csv'),
                                index=False, encoding='utf-8-sig')
print("  → low_risk_high_camera.csv 저장")

# 재배치 요약
redistribution = pd.DataFrame({
    '구분': ['과잉 구역', '적정 구역', '부족 구역', '위험 구역'],
    '설명': ['카메라 감축 가능', '현행 유지', '카메라 증설 필요', '카메라 효과 점검 필요'],
    '해당 시도': [
        ', '.join(과잉_list) if 과잉_list else '-',
        ', '.join(적정_list) if 적정_list else '-',
        ', '.join(부족_list) if 부족_list else '-',
        ', '.join(위험_list) if 위험_list else '-'
    ]
})
redistribution.to_csv(os.path.join(OUTPUT_DIR, 'redistribution_summary.csv'),
                      index=False, encoding='utf-8-sig')
print("  → redistribution_summary.csv 저장")


# ── 최종 콘솔 요약 ────────────────────────────────────────────
print("\n" + "=" * 70)
print("  분석 완료!")
print("=" * 70)
print(f"\n  📁 출력 폴더: {os.path.abspath(OUTPUT_DIR)}")
print(f"  📊 차트 파일: 5개 PNG")
print(f"  📄 데이터 파일: 4개 CSV")
print(f"\n  🔴 카메라 부족 → {', '.join(부족_list) if 부족_list else '해당 없음'}")
print(f"  🔵 카메라 과잉 → {', '.join(과잉_list) if 과잉_list else '해당 없음'}")
print(f"  💡 재배치 방향: 과잉 구역 카메라 → 부족 구역으로 이동")
print()

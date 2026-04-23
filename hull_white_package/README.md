# Hull-White V4 Final — 信用模型市场基准评估

MF728 Group Project 的 Hull-White 子模块。V4 把 "预测 default_label" 的离散任务替换为 "预测 Markit CDS 隐含 hazard rate λ" 的连续任务，让 walk-forward OOT 评估自然可行。本目录是最终交付包。

---

## 目录结构

```
hull_white_package/
├── README.md                                  # 本文件
├── notebooks/
│   └── hw_cds_benchmark_v4.ipynb              # 主分析 notebook（端到端可运行）
├── outputs/
│   ├── v3_shumway/                            # HW 训练用的 V3 Shumway 面板
│   │   ├── panel_shumway_v3.csv               # 162 firms × 2021-2023，含 20 个 Shumway 正例
│   │   ├── metrics_v3.csv                     # V3 Stage A (concurrent) in-sample 指标
│   │   └── metrics_lagged_v3.csv              # V3 Stage B2 (lagged) in-sample 指标
│   └── v4_final/                              # Round 1-5 冻结产物（V4 主成果）
│       ├── panel_v4_final.csv                 # 最终面板（包含所有 λ/PD 列）
│       ├── final_comparison.csv               # 单张对比总表
│       ├── final_taskB_heatmap.png            # Task B 判别力热图
│       ├── final_taskC_heatmap.png            # Task C 对 avrating 热图
│       ├── final_taskB_values.csv             # Task B 原始值
│       ├── final_taskC_values.csv             # Task C 原始值
│       ├── final_calibration.csv              # walk-forward 线性校准结果
│       ├── final_sensitivity.csv              # Recovery R 与 IV lag 敏感性
│       └── round_log.txt                      # Round 1-5 全部关键数字
└── report/
    ├── Hull_White_V4_Final_Review.docx        # 主报告（5 轮审核结论）
    ├── Hull_White_V4_Final_公式介绍.docx      # 公式推导
    ├── Hull_White_V4_Final_公式与问题速览.docx # Q&A 速查
    └── OFFICE_HOUR_walkthrough.md             # 25 分钟演讲脚本
```

---

## 对父项目的依赖

本包是 `merton-main` 项目的子模块，notebook 从项目根目录运行时自动读取：

| 路径 | 用途 |
|---|---|
| `config.py` | 全局参数（T=1.0，RECOVERY_RATE=0.4 等） |
| `models/merton.py` | Merton 结构化模型 |
| `models/vk.py` | Vasicek-Kealhofer 模型 |
| `models/hull_white.py` | HW reduced-form intensity 模型 |
| `data/implied_vol.py` | OptionMetrics IV 加载器 |
| `data/cds_parspread.csv` | Markit 5Y CDS spread + dp + avrating（84 firms × 5 yr） |
| `data/implied_vol_optionmetrics.csv` | 增强 equity_vol 的隐含波动率 |
| `output/panel_data.csv` | V3 已缓存的基础面板（163 firms × 5 yr） |

---

## 如何运行

### 1. 环境准备（在项目根目录执行）

```bash
pip install numpy pandas scipy scikit-learn matplotlib jupyter
```

### 2. 运行 notebook

```bash
cd /path/to/merton-main
jupyter nbconvert --to notebook --execute \
    hull_white_package/notebooks/hw_cds_benchmark_v4.ipynb \
    --inplace --ExecutePreprocessor.timeout=600
```

或者在 Jupyter 里打开 `hull_white_package/notebooks/hw_cds_benchmark_v4.ipynb` 逐单元执行。

> Notebook 第 1 单元有 `while not os.path.exists('config.py'): os.chdir('..')`，会自动向上找到项目根目录再 `sys.path.insert`，所以无论从哪个工作目录启动 Jupyter 都能正确导入。

### 3. 输出位置

Notebook 运行时会在 `hull_white_package/outputs/v4_cds/` 下生成：

- `panel_v4.csv`（314 行 V4 panel，带 Merton/VK/HW 全部 λ 与 PD 列）
- `taskA_level_matching.csv`、`taskA_calibration.csv`
- `taskB_rank_values.csv`、`taskB_rank_heatmap.png`
- `taskC_rating_values.csv`、`taskC_rating_heatmap.png`
- `final_comparison.csv`

冻结版本（含 5 轮审核的敏感性产物）已预先存放在 `outputs/v4_final/`，供报告直接引用。

---

## 三大评估 Task

| Task | 对比 | 关键指标 | 说明 |
|---|---|---|---|
| **A** | `λ_model` vs `λ_market` | R² / RMSE / Spearman | level 匹配；IG 公司结构模型系统性低估 |
| **A'** | walk-forward calibration | 线性 `a + b·λ_model` | train ≤ k → test k+1（k∈{2022,2023,2024}） |
| **B** | 每年横截面排序 | per-year Spearman | 无训练、无 horizon、无 recovery，最纯判别度量 |
| **C** | 模型分数 vs S&P avrating | Spearman | 与 Markit impliedrating head-to-head |

---

## 5 轮审核核心发现（详见 `report/Hull_White_V4_Final_Review.docx`）

1. **模型差异不显著**：Task A 95% firm-level bootstrap CI 全部重叠（Merton 0.567 [0.43, 0.67] / VK 0.556 [0.42, 0.66] / HW 0.526 [0.37, 0.66]）。
2. **HW transfer 无偏**：out-of-firm vs in-sample Spearman 差 0.0003，V3 → V4 迁移不是问题来源。
3. **t-1 IV 反直觉更强**：HW pooled Spearman 0.526 → 0.549 (+0.023)，RMSE 降 30%（0.115 → 0.081）。
4. **Recovery R 几乎无影响**：R ∈ {0.3, 0.4, 0.5} 下 Spearman 不变，校准系数 b 线性偏移。
5. **log-RMSE 反转排名**：log 空间下 HW lagged=1.59 远优于 Merton 12.78 / VK 13.33；绝对 RMSE 因 HW level 偏差反而垫底。

---

## 核心 Limitations

1. 样本全为投资级（84 firms 均在 A/AA/BBB），tail risk（HY、default）未验证。
2. `dp` 与 `parspread` 数学等价（都是 CDS spread 的函数），预测 dp == 预测 spread。
3. HW 系数是从 V3 的 162 firm 面板迁移过来；V4 IG universe 无正例，无法就地重训。
4. 2021 年缓存面板仅 5 行，实质评估从 2022 开始。
5. V3 附录 AUC（0.975 / 0.994）**不是** out-of-time，是 within-2023 in-sample，报告时已显式标注。

---

## 引用路径

演讲/答辩时手边打开：

- `outputs/v4_final/final_comparison.csv` — 单表对比
- `outputs/v4_final/final_taskB_heatmap.png` / `final_taskC_heatmap.png` — 热图
- `outputs/v4_final/round_log.txt` — Round 1-5 关键数字
- `report/Hull_White_V4_Final_Review.docx` — 主报告
- `report/OFFICE_HOUR_walkthrough.md` — 25 分钟演讲脚本

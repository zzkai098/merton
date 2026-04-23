# Office Hour Walk-Through — Hull-White V4-Final

**目标会议时间**：2026-04-22（明天）

**手边要打开**：
- `hull_white_package/outputs/v4_final/final_comparison.csv`
- `hull_white_package/outputs/v4_final/final_taskB_heatmap.png`
- `hull_white_package/outputs/v4_final/final_taskC_heatmap.png`
- `hull_white_package/report/Hull_White_V4_Final_Review.docx`（主报告）
- `hull_white_package/report/Hull_White_V4_Final_公式介绍.docx`（公式推导）

---

## 25 分钟流程

### 0–3 分钟｜项目主线（开场）
> "这个项目是 MF728 group project。我们对比三个信用风险模型：Merton (1974)、Vasicek-Kealhofer (KMV)、Hull-White reduced-form。
> V1-V3 用 default_label 二分类做评估。但 config.py 把 21 家 distressed firms 的 distress_date 都硬编码为 2023-12-31，
> 正例集中在单年，walk-forward 训练集永远 0 正例，跑不起来。所以 V3 报的 AUC 0.975 / 0.994 是 within-2023 in-sample，不是 OOT。
> V4 做了范式转变：换成预测 Markit CDS 隐含 hazard rate λ。84 家 IG firms × 5 年 = 314 行，连续目标，walk-forward 天然可行。
> 这次 office hour 我想把 V4 之后做的 5 轮审核结果呈给您看一下。"

### 3–7 分钟｜V4 baseline 三个 task 速览
打开 V4 baseline 报告 §6.3：
- **Task A**: λ_model vs λ_market 的 RMSE / R² / Spearman（pooled across years）
- **Task A'**: walk-forward calibration `λ_market = a + b·λ_model`（train ≤ k → test k+1）
- **Task B**: per-year 横截面 Spearman（无训练，最干净的判别度量）
- **Task C**: 把模型分数对 S&P/Moody's avrating 排序

> "V4 的核心结论是：三个模型都能 rank IG 公司风险，但 level 上系统性低估（Task A R² 都是 -2 到 -290）。
> calibration 之后能改善但仍然 R² < 0。Task C 上 HW 看起来最强（Spearman 0.63），但是 V4 写的 limitations 里
> 已经承认这不是公平比较（HW 的特征和评级机构输入对齐）。"

### 7–22 分钟｜Round 1-5 五个核心发现

打开 V4 Final Review docx 的 §3 一节一节走，或者直接用下面这个速记板：

#### 发现 1（Round 1）：模型差异统计上不显著
打开 `final_comparison.csv`，指着 Task A 的 95% CI 列：
- Merton: 0.567 [0.43, 0.67]
- VK    : 0.556 [0.42, 0.66]
- HW lagged: 0.549 [0.40, 0.67]
- HW concurrent: 0.526 [0.37, 0.66]

> "这些 CI 是 firm-level bootstrap (B=1000)，按 ticker 重抽样。所有 CI 互相重叠，说明这些 0.04 量级的差距在 n=314、ticker 高度重叠的 IG 样本上**不能**被 bootstrap 区分。报告应该写'rank 能力相当'，不写'某模型最强'。"

#### 发现 2（Round 2）：HW transfer 不是问题
> "V4 limitations 第 3 条担心 HW 的训练集 (V3 162 firms) 严格包含 V4 的 84 IG firms，所以是 firm-level in-sample。
> 但 V3 的 20 个 Shumway 正例（HTZ 除外的 distressed firms）**没有一个**在 V4 的 IG 集合里——Markit 不为 distressed firms 收录 CDS。
> 所以我们可以训练'V3 减 V4 firms'的 79 firm 模型，20 正例全部保留，0 firm overlap。"
> 
> "结果：HW out-of-firm vs in-sample 的 fair (22-25) Spearman 差 **0.0003**。可以把这条 limitation 拿掉。"

#### 发现 3（Round 3，反直觉）：t-1 IV 比 concurrent IV 更好
> "V4 limitations 第 9 条担心 IV 同期泄漏：OptionMetrics IV 是 12月末观察的 forward-looking 1Y vol，V3 事件年是 2023-12-31，
> IV 等于把已知 distress 信号喂进去。我们试着把 V3 训练集和 V4 应用集都换成 t-1 年 IV。"
> 
> "**反直觉发现**：HW pooled Spearman 从 0.526 升到 **0.549** (+0.023)，RMSE 从 0.115 降到 **0.081** (-30%)。"
> 
> "我的理解：concurrent IV 把 2023 distress 信号过强地拉进训练集，分界面 over-fit 到 2023 这个 regime；
> lagged IV 让训练分布更接近 V4 panel 中 IG firms 的正常 IV 分布，generalize 更好。
> 这是经典的'数据分布一致性 > 数据信息量'，但在信用模型 context 下出现，值得记一笔。"

#### 发现 4（Round 3）：Recovery R 敏感性
> "Markit dp 内部用 R ≈ 0.4。我用一阶恒等式 λ(R) = spread / (1-R) 重算 R ∈ {0.30, 0.40, 0.50}：
> Spearman 完全不变（R 是 λ 的单调线性变换）；calib b 系数按 1/(1-R) 线性变化（Merton: R=0.3→b≈0.43, R=0.4→0.50, R=0.5→0.60）；
> calib_R² 完全不变。所以**所有 rank 度量对 R robust，level 度量随 R 线性偏移**。报告 b 时要带 R 标注。"

#### 发现 5（Round 4）：log-RMSE 完全反转 RMSE 排名
> "λ 跨 5 个数量级：Merton/VK 对 IG 公司预测 1Y PD ≈ 1e-15，市场 λ ≈ 0.008。Plain RMSE 被 level 主导。
> log-RMSE：Merton **12.78** / VK 13.33 / HW concurrent 1.82 / HW lagged **1.59**。
> 在对数空间下 HW 完胜。结构化模型把 IG PD 预测到 1e-15 量级，log 距离 ~30，几何上不合理。
> **做 level 比较时应优先报 logRMSE**。"

#### 副发现（Round 4）：walk-forward window 必须 expanding
> "V4 用 expanding window (n=78/156/235)。我们试 rolling 2y / rolling 1y。
> rolling 1y 在 2023→2024 split 时 Merton b = **87**、VK b = **347**，回归崩。
> 原因：单年 IG 公司 λ 集中（年内 std ≈ 0.001），X 方差极小，b = cov/var 数学上爆。
> 所以 V4 的 expanding 选择并非任意，是必选。"

### 22–25 分钟｜主动提出讨论问题（见下文 Q1-Q7）

---

## 你可能被问到的问题（10 道）

### Q1：为什么不直接重训练 HW 在 IG panel 上？
A：V4 的 84 家 IG panel 中 default_label 全是 0（无正例），logistic 触发 hull_white.py:110-119 单类 fallback，hazard 全部输出 0。所以用 V3 的 162-firm Shumway panel 训练后 transfer。Round 2 已验证 transfer 不引入偏差。

### Q2：t-1 IV 比 concurrent IV 表现好不奇怪吗？
A：奇怪。解释：concurrent IV 把 2023 distress 信息间接喂给模型，分界面 over-fit 到 2023 regime；lagged IV 让训练分布与 V4 IG 分布更一致，generalize 更好。是 ML 里"分布一致性 > 信息量"的经典例子。

### Q3：Task A R² = -290 是不是说明 HW 完全没用？
A：不是。R² 在 level 系统偏差下不合适。看 (a) Spearman 0.55 → rank 合理；(b) log-RMSE 1.59 vs Merton 12.78 → 对数空间下 HW 最好；(c) calibrated R²（2024→2025 split）HW +0.20，Merton -0.26，HW 是唯一在 OOT split 上 R² > 0 的模型。

### Q4：为什么 Merton 和 VK 的 calib b 差 3 倍？
A：default_point 定义不同。Merton 用 total_debt；VK 用 STD + 0.5×LTD（约 60-70% of total_debt）。VK 的 V/dp 比例更大 → DD 更大 → PD 更小 → λ 更小 → 校准时需要更大的 b 放大。Round 2 验证：换 measure (mu=r vs mu=r+0.02) 只能让 b 抬高 18%，不能弥合 3 倍差距。是模型结构差异。

### Q5：你的样本只有 IG，怎么知道在 HY 上的表现？
A：不知道。这是 V4 报告 §9 第 1 条 limitation，V4-final 没有改善。Markit 不为 distressed names 收录 CDS。后续方向：(a) Bloomberg HY CDS；(b) KMV EDF proxy；(c) 5Y bond yield − Treasury surrogate spread。

### Q6：V3 的 0.975 AUC 真的没用吗？
A：V3 AUC 衡量"in-sample within-2023 cross-validation"的 rank 能力——能找到判别面，但不能预测未来未知的 distress。更严重的是 walk-forward 失败不是 data scarce，是 label by construction concentrated（config.py 硬编码 2023-12-31）——再加 5 年数据正例仍全在 2023。V4 改换评估对象绕开这个结构性问题。

### Q7：Task C 上 HW 比 Merton/VK 强这么多怎么解释？
A：**不是公平比较**。HW 用的特征 (leverage, equity_vol, log_market_cap) 正是评级机构输入；Merton/VK 用的是结构化合成的 DD（一个数字）。HW 的优势反映"输入信号与评级机构对齐"，不是"HW 模型本身更强"。要公平对比，应该把 Merton DD + HW 三特征放进同一 logistic 看 incremental contribution。

### Q8：bootstrap CI 那么宽，为什么还要报点估计？
A：点估计是中心，CI 是不确定性。用 firm-level bootstrap（不是 row-level）因为同 firm 跨年观测高度相关。CI 宽因为 (a) n=314 偏小；(b) 83 ticker 在 5 年里高度重复，等价独立 firm 数 ~83。所以核心结论应该定性（"三个模型 rank 能力相当"）而不是定量。

### Q9：为什么修了 hw_lambda 公式后 Task B Spearman 没变？
A：Task B Spearman 是 rank 测度，对单调变换 (hazard ↔ λ ↔ PD) 不敏感。修正只影响 Task A 的 RMSE/R²（level 类）。这正佐证 Task B 设计意图：排除任何单调变换 + recovery + horizon 的影响，只测 rank。

### Q10：5 轮里哪个发现最意外？
A：Round 3 的 IV lag 结果。一开始假设 t-1 IV 会让 HW 表现下降，实际反而 Spearman + 0.023, RMSE - 30%。后来理解为"分布一致性 > 信息量"。这是一个值得在最终报告里 highlight 的发现。

---

## 你可以主动问教授的问题（7 道）

### A. 报告设计层面
1. **R² 还是 logRMSE？** Task A R² 全部为负、log-RMSE Merton 12.78、Spearman 0.56。对 IG-only panel 报告 level 类指标本身是不是没意义？是不是应该只报 rank + calibration b？

2. **范式转变是否还在 task 范围？** V3 → V4 等于换了 ground truth（default_label → market λ）。这是否还在 "Hull-White 模型评估" 范围内？还是已偏到 "CDS 反推校准"？

3. **Task C 的 feature alignment 声明？** HW 用三特征直接对 avrating 排序等于把 HW 当 rating-replication model 用。是否需要在报告里加一个 "feature alignment" 声明？

### B. 方法论层面
4. **firm-level vs ticker-block bootstrap？** 我们用 firm-level bootstrap (B=1000) 算 CI 发现差异不显著。如果你写论文，会用 firm-level 还是 ticker-block，或者别的？

5. **OOT 评估改成 first-difference？** rolling 1y 失败说明 IG 单年 λ 太集中。是否应该把目标换成 "λ 的年度变化" 而不是 λ 水平？（first-difference 形式可能减小自相关）

6. **lagged-IV 发现的写法？** 我们改 lagged-IV 后 HW 表现反而更好。这是值得作为 main result 写还是作为 sensitivity 放附录？

### C. 数据层面
7. **真实 distress 时间点怎么收集？** 21 家 firms 的 distress_date 都硬编码 2023-12-31。如果想做真正 OOT default classifier，应该如何收集真实 distress 时间？Chapter 11 申请日 / rating downgrade 日 / going-concern opinion 日，哪个最 academic acceptable？

---

## 速查：关键数字一览（office hour 桌上贴一张）

| 指标 | Merton | VK | HW concurrent | HW lagged-IV |
|---|---|---|---|---|
| Task A pooled Spearman | 0.567 | 0.556 | 0.526 | **0.549** |
| Task A 95% CI lo | 0.428 | 0.422 | 0.371 | 0.402 |
| Task A 95% CI hi | 0.675 | 0.656 | 0.660 | 0.675 |
| Task A logRMSE | 12.78 | 13.33 | 1.82 | **1.59** |
| Task A RMSE (plain) | **0.0117** | 0.0118 | 0.115 | 0.081 |
| Task B fair (22-25) Spearman | **0.558** | 0.545 | 0.498 | 0.522 |
| Task A' calib R² (2024→2025) | -0.26 | -0.26 | +0.11 | **+0.20** |
| Task C fair (22-25) avg | 0.416 | 0.388 | 0.617* | 0.626* |

\* HW Task C 的 0.62 数字带特征对齐 caveat（见 Q7）。

| Round 编号 | 主要发现 | docx 章节 |
|---|---|---|
| R1 | bootstrap CI / hw_lambda fix / fair 口径 → 模型差异不显著 | §3.1 |
| R2 | HW out-of-firm = in-sample (delta 0.0003) | §3.2 |
| R2 | VK measure 切换不能解释 b 差异（结构差异是主因） | §3.2 |
| R3 | **lagged-IV 比 concurrent-IV 好** (Spearman +0.023, RMSE -30%) | §3.3 |
| R3 | Recovery R 敏感性：rank invariant, b 线性偏移 | §3.3 |
| R4 | walk-forward 必须 expanding，rolling 1y b 爆掉 | §3.4 |
| R4 | **log-RMSE 完全反转 RMSE 排名** (HW 1.6 vs Merton 12.8) | §3.4 |
| R5 | 整合冻结到 outputs/v4_final/ | §3.5 |

---

## V4 limitations 状态对照（office hour 末尾自查）

| V4 limitation §9 | Round | 状态 |
|---|---|---|
| #1 全 IG 样本 | — | **未解决**（需要 HY 数据） |
| #2 dp 与 spread 数学等价 | — | **未解决**（同一信息源） |
| #3 HW transfer in-sample | R2 | ✅ 已检验，OOF 与 in-sample 几乎相同 |
| #4 IV concurrent 泄漏 | R3 | ✅ 已修；t-1 IV 反而更好 |
| #5 Measure mismatch | R2 | ✅ 已检验；不能解释 Merton-VK b 差异 |
| #6 Walk-forward 训练窗口递增 | R4 | ✅ 已检验；rolling 1y 崩，expanding 是必选 |
| #7 Task B HW mean 口径不一致 | R1 | ✅ 已修；引入 fair_22_25 列 |
| #8 Task C HW 特征对齐评级 | — | **未修**（需要更多 baseline 模型） |
| #9 OLS 写法 | R1 | ✅ 已改 np.polyfit |
| #10 hw_lambda = hw_hazard | R1 | ✅ 已修；用 -log(1-PD)/T |
| #11 缺 bootstrap CI | R1+R4 | ✅ 已加 firm-level B=1000 |
| #12 Recovery sensitivity | R3 | ✅ 已加 R∈{0.3,0.4,0.5} 表 |

剩余 3 条无法在当前数据/范围内解决，office hour 最后跟教授沟通后续方向。

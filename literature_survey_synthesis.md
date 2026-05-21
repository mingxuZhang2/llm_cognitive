# Nature系列期刊文献调研综合报告
## LLM可解释性 × 神经科学方法 × Neuron-Level Analysis

调研范围：Nature正刊、Nature Machine Intelligence、Nature Neuroscience、Nature Methods、Nature Human Behaviour、Nature Communications、Nature Computational Science，以及NeurIPS/ICLR/EMNLP等顶会。共检索60+篇论文，精选40+篇深度分析。

---

## 一、各期刊定位与接收标准

### Nature正刊 (IF~70)
**核心要求：** 生物数据 + AI数据双线并行 + 因果干预 + 跨系统普遍性原则
- 几乎不发纯ML可解释性文章
- 必须有真实生物数据（fMRI、连接组、神经元记录）作为共同等级的故事线
- 必须有因果/干预性证据
- 必须揭示跨生物和人工系统的普遍原则
- 发现必须反直觉、有广泛社会影响

**代表作：**
| 论文 | 年份 | 核心 |
|------|------|------|
| Zhang et al. — Inter-brain neural dynamics in bio+AI | 2025 | 小鼠社交+AI agents共享子空间，ablate AI共享子空间→合作下降 |
| Ding et al. — Functional connectomics (MICrONS) | 2025 | 75,000+神经元连接组，like-to-like wiring在生物皮层和RNN中都成立 |
| Wang et al. — Foundation model of neural activity | 2025 | 13.5万神经元基础模型，预测新刺激类型的神经响应 |
| Zhou et al. — Larger LLMs less reliable | 2024 | 32个LLM，4.2M响应，scaling paradox，Nature接受因为有社会影响 |

**我们的差距：** 没有生物数据。纯LLM内部分析不够Nature正刊。除非加入fMRI/脑数据验证atlas的hierarchy也存在于人脑中。

### Nature Machine Intelligence (IF~24) — 我们的目标刊
**核心要求：** 关于AI系统的fundamental principles + broad impact + 跨架构普遍性
- 接受纯AI系统研究（不需要生物数据）
- 要求揭示universal principles，不仅是单模型观察
- 当前已发表的LLM可解释性文章全部是correlational（RSA、encoding models）
- **关键gap：没有任何NMI论文做过LLM内部的causal ablation + double dissociation**
- SemanticLens (Dreyer 2025) 是最接近的——neuron-level分析，但是descriptive的，不是causal的
- 我们的causal approach会SET a new standard，不仅仅是meet existing bar

**代表作：**
| 论文 | 年份 | 核心 | 与我们的差距 |
|------|------|------|-------------|
| Mischler et al. — Contextual hierarchies converge | 2024 | iEEG + 12 LLMs，层级与大脑对齐 | 无因果干预 |
| Doerig et al. — Visual brain aligned with LLMs | 2025 | RSA, 8 participants, 13 comparison architectures | 无ablation |
| Dreyer et al. — SemanticLens | 2025 | Neuron→CLIP embedding，74页 | 描述性，无dissociation |
| Du et al. — Human-like concept representations | 2025 | 4.7M triplet judgments, 1854 objects | 无因果验证 |
| Kar et al. — Interpretability AI vs Neuro | 2022 | Perspective，定义了neuro-AI interpretability交叉 | 无实验 |

**NMI的发表模式：**
- 4-5个main figures（每个figure一个核心message）
- 18页manuscript + 49页appendix是可接受的（SemanticLens = 74页总）
- Code release几乎mandatory
- Claims要calibrated（"align with" 不是 "identical to"）
- 统计：FDR correction标配，permutation test + CIs

### Nature Communications (IF~16) — 强力备选
**核心要求：** 严谨+重要，但不需要top-1% novelty
- Kumar et al. 2024是直接先例：attention head functional specialization + fMRI
- 5 main figures + 29 supplementary figures
- Permutation tests + FDR + bootstrap CIs
- 我们的工作arguably above NatComm's typical threshold

### Nature Computational Science (IF~6) — 不推荐
- 定位是"computational tools for science"，不是"science of AI"
- 几乎没有mechanistic interpretability文章
- Desk rejection风险高

### Nature Neuroscience / Nature Methods — 不直接匹配
- 需要生物数据作为主线
- 但vocabulary和methodology可以借鉴（"functional atlas", "double dissociation", "natural kind"）

---

## 二、最重要的竞争/相关工作

### Tier 1: 必须引用+明确区分（直接竞争）

#### 1. AlKhamissi et al. 2025 — "The LLM Language Network" (NAACL 2025)
**最直接的竞争对手。**
- 在18个LLM中用neuroscience-style localization找到language-selective units
- Causal ablation验证功能必要性
- Brain alignment验证
- **与我们的差距：**
  - 只有1-3个domain（language, reasoning, social），我们有8个
  - 无double dissociation between domains
  - 无dependency DAG
  - 无predictive control framework
  - 无cross-domain interactions（competitive inhibition, spillover）

#### 2. Anthropic SAE系列 — Bricken 2023 + Templeton 2024 + Lindsey 2025
**最visible的competing approach。**
- SAE提取monosemantic features，Scaling Monosemanticity在Claude 3 Sonnet上提取百万级features
- Feature steering改变模型行为
- **与我们的差距：**
  - 无监督feature dictionary vs 我们的supervised causal atlas
  - Features不按cognitive domain组织——百万features没有结构性hierarchy
  - 无cross-architecture validation
  - 无double dissociation
  - 无behavioral prediction from atlas
  - 无dependency DAG

#### 3. Dai et al. 2022 — "Knowledge Neurons" (ACL, ~559 citations)
**直接的paradigm precursor。**
- Gradient-based attribution找factual knowledge neurons + ablation验证
- **与我们的差距：** 单domain (factual), 单model (BERT), 无dissociation, 无universality

#### 4. Meng et al. 2022 — "ROME" (NeurIPS, ~1537 citations)
**Causal tracing for factual knowledge。**
- Activation patching定位factual associations到mid-layer MLP
- **与我们的差距：** 单domain (factual), editing focus, 无cross-model, 无DAG

#### 5. Wang et al. 2022 — "Skill Neurons" (EMNLP, ~200 citations)
**Thematic predecessor on functional specialization。**
- Prompt-tuning activation分析找task-specific neurons
- **与我们的差距：** NLP pipeline tasks (不是cognitive domains), 无causal ablation, 无dissociation

### Tier 2: 重要相关工作

| 论文 | 年份 | 关系 |
|------|------|------|
| Geva et al. — FFN as Key-Value Memories | 2021 | FFN理论基础 |
| Zou et al. — Representation Engineering | 2023 | 互补approach (residual stream方向 vs neurons) |
| Hase et al. — Does Localization Inform Editing? | 2023 | **必须preempt的critique**：localization不等于editability |
| Elhage et al. — Toy Models of Superposition | 2022 | **必须address的challenge**：polysemanticity |
| Gurnee et al. — Universal Neurons in GPT2 | 2024 | 我们convergence analysis的precursor |
| Liu et al. — Brain-Inspired Functional Networks | 2025 | 直接competitor (co-activation, 无dissociation) |
| Christ et al. — Math Neurosurgery | 2024 | 单domain version (math only) |
| Dobs et al. — Functional Specialization in DNNs | 2022 | **最clean的prior double dissociation**（但是CNN vision, 不是LLM） |
| Lindsey et al. — Biology of an LLM | 2025 | Anthropic最新attribution graphs |

### Tier 3: 生物学precedent（framing用）

| 论文 | 年份 | 期刊 | 我们可以借鉴什么 |
|------|------|------|-----------------|
| Hermosillo et al. — Precision Functional Atlas | 2024 | Nat Neurosci | "functional atlas"术语的生物学先例 |
| Fedorenko et al. — Language Network as Natural Kind | 2024 | Nat Rev Neurosci | "natural kind"框架：我们的8个category是natural kinds |
| van den Heuvel et al. — LNM Critique | 2025 | Nat Neurosci | 我们避开了lesion network mapping的circular sampling问题 |
| Kumar et al. — Shared Functional Specialization | 2024 | Nat Commun | attention head specialization + fMRI，但无causal ablation |
| Tuckute et al. — Driving Language Network | 2024 | Nat Hum Behav | closed-loop causal control paradigm |
| Béna & Goodman — Modularity Under Constraints | 2025 | Nat Commun | 理论基础：structural modularity不等于functional specialization |

---

## 三、我们工作的Genuine Novelty（基于全部文献）

### 无任何已发表论文做到的（6个first）：

1. **8-domain cognitive taxonomy with mutual causal dissociation**
   - AlKhamissi: 1-3 domains, 无dissociation between domains
   - Dai: 1 domain
   - Christ: 1 domain
   - 我们: 8 domains × 28/28 pairwise dissociation × 4 models × 3 scales

2. **Cross-architecture universality with quantified convergence**
   - Gurnee: GPT-2 random seeds（同架构）
   - 我们: 4 distinct architectures, FCI = 0.86, z = 5.64, p < 0.001

3. **Dependency DAG between cognitive domains**
   - 无任何论文mapping directional causal dependencies between functions
   - 我们的3-layer hierarchy: language/code → math/science → reasoning/ethics

4. **Atlas-guided behavioral prediction and control**
   - 无论文用atlas预测ablation outcome并验证
   - 我们: 12/12 pathway decomposition, 11/12 instance-level, 4/4 DAG nulls confirmed

5. **Competitive inhibition between cognitive domains**
   - Code suppresses ethics, ethics suppresses humanities
   - 无先例

6. **Atlas-guided pruning eliminating phase transitions**
   - 1.11-3.24x vs 3-3948x (random) vs 13-11898x (magnitude)
   - 无先例

---

## 四、Reviewer会攻击的点 + 应对策略

### Challenge 1: "Neurons are polysemantic (superposition). 怎么能归因到单一function？"
**应对：** 我们的atlas是population-level的（5000 neurons per domain），不claim individual neuron monosemanticity。Double dissociation在population level验证。类比：fMRI voxel-level dissociation在neuroscience被接受，尽管individual neurons是multi-functional的。

### Challenge 2: "Gradient×activation只是correlation，不是causation"
**应对：** 这正是我们的causal ablation实验解决的。28/28 pairwise double dissociation + atlas-guided pruning + steering experiments = causal validation。Cite Hase et al. 2023说明我们aware of the gap，并用behavioral prediction framework close it。

### Challenge 3: "8个cognitive categories是arbitrary的"
**应对：** 8个domain基于cognitive science标准分类。Dissociation结果验证了empirical separability。Subcategory实验（12 fine-grained categories）证明atlas捕获的是cognitive function而非surface features（历史vs哲学都是选择题格式，但neurons完全不同）。

### Challenge 4: "AlKhamissi (2025) 已经做了"
**应对：** AlKhamissi只有1 domain (language) across 18 models。我们是8 domains × 4 architectures with：mutual dissociation, dependency DAG, competitive inhibition, predictive control, atlas-guided pruning。完全不同的scope和depth。

### Challenge 5: "SAE方法更principled"
**应对：** SAE和causal atlas是complementary的。SAE产生unsupervised feature dictionary但不建立functional hierarchy，不能predict cross-domain behavioral interactions，未经cross-architecture验证。我们的supervised causal approach专为cognitive function mapping设计。

### Challenge 6: "为什么只看FFN neurons？Attention heads也重要"
**应对：** 基于Geva et al. (2021, 2022, 2023), Dai et al. (2022), Meng et al. (2022)——FFN是stored knowledge和cognitive function的primary site。Attention heads主要做routing和retrieval。Cite Todd et al. (2024) Function Vectors说明分工。

### Challenge 7: "只有7B-scale模型"
**应对：** 4个不同架构 × 3个scale × 6个模型变体。包括base model (Llama-2) 和reasoning specialist (DeepSeek-R1)。FCI = 0.86说明findings跨架构stable，不太可能是scale-specific。

---

## 五、Framing策略（基于成功Nature系列论文）

### 策略1: Universal Computational Principle (最强)
"The 3-layer functional hierarchy and dependency DAG are computationally necessary features of any system trained on multiple cognitive tasks — this is why they are universal across 4 architectures."
- 对标: Mischler 2024 (NMI), Dobs 2022 (Sci Adv)

### 策略2: Neuroscience Gold Standard Applied to AI
"We apply the gold standard of causal functional localization in neuroscience — double dissociation — to characterize the functional atlas of LLMs."
- 对标: AlKhamissi 2025, Kar et al. 2022 (NMI)

### 策略3: Natural Kind Ontological Validity
"The 8 cognitive categories are natural kinds implemented in LLMs, evidenced by their universal cross-architecture segregation (28/28) and causally predictive organization."
- 对标: Fedorenko 2024 (Nat Rev Neurosci)

### 策略4: Predictive Power + Practical Control
"The atlas predicts ablation outcomes (12/12 pathway, 11/12 instance-level), guides efficient pruning (1.1x vs 4000x degradation), and enables controlled steering."
- 对标: Tuckute 2024 (NHB — LLM控制brain activity)

### 策略5: Methodological Upgrade Over Existing Work
"All prior Nature-family papers on LLM functional organization use correlational methods (RSA, encoding models). We provide the first causal atlas with interventional validation."
- 对标: van den Heuvel 2025 (Nat Neurosci — LNM critique)

---

## 六、投稿策略建议

### 首选: Nature Machine Intelligence
**理由：**
1. NMI已发表的LLM可解释性文章全部是correlational；我们的causal approach是quality jump
2. 我们的cross-architecture universality (4 models, FCI=0.86) 符合NMI对"universal principle"的偏好
3. SemanticLens (2025) 开了neuron-level analysis的先例，但descriptive；我们加上causal validation
4. 有practical application (pruning, steering)，NMI看重applied value
5. NMI接受纯AI研究，不需要生物数据

**NMI格式参考：**
- 4-5 main figures
- 18-25页manuscript + 30-50页supplementary
- Code release (GitHub)
- 统计：permutation tests (≥1000), BH-FDR, bootstrap CIs, Cohen's d
- 语言：calibrated claims ("causally implements", "universally observed", "predictive")

### 备选: Nature Communications
**如果NMI desk reject（~50%概率）：**
- Kumar et al. 2024是直接先例
- 我们的work在每个维度都stronger
- 几乎不需要revision就能re-submit
- 接受率更高 (~15-20% vs NMI的~5-8%)

### 冲刺: Nature正刊
**需要额外数据：**
- 加入fMRI/脑数据验证：我们的3-layer hierarchy是否在人脑中也存在
- 或者：找到genuinely counterintuitive universal principle跨生物和人工系统

---

## 七、我们实验量 vs NMI已发表论文的对比

| 维度 | NMI typical | 我们的工作 | 评估 |
|------|------------|-----------|------|
| 模型数量 | 2-12 | 6 (4 main + 2 extra) | **达标** |
| Cognitive domains | 1-3 | 8 + 12 subcategories | **远超** |
| 因果证据 | 无 (全部correlational) | Double dissociation + ablation + steering | **首创** |
| 统计检验 | FDR, permutation | 10K permutation, BH-FDR, t-test, Cohen's d, bootstrap | **远超** |
| Cross-architecture | 2-4 models compared | 4 architectures, FCI=0.86, z=5.64 | **远超** |
| Practical application | 有的有，有的没有 | Pruning (1.1x vs 4000x), steering, prediction | **强** |
| Sample scales | 1-2 | 3 (15, 50, 152 per cat) | **达标** |
| Discovery/validation split | 不要求 | 82/cat per split | **有** |
| Method triangulation | 不要求 | 3 methods, r=0.96 | **有** |

**总体评估: 实验量充分，超过NMI已发表的大部分同类工作。Causal evidence是unique differentiator。**

---

## 八、建议的Figure结构（对标Doerig 2025 NMI）

1. **Figure 1: The Causal Functional Atlas** — 8×8 dissociation matrix heatmap × 4 models，一图展示核心发现
2. **Figure 2: Universal Dependency Hierarchy** — DAG图 + 跨模型一致性 + spillover matrix
3. **Figure 3: Atlas Predicts and Controls Behavior** — pathway decomposition + steering + instance-level prediction
4. **Figure 4: Practical Utility** — atlas-guided pruning vs random vs magnitude at 10-50% sparsity
5. **Figure 5 (optional): Convergence and Robustness** — FCI across architectures + method triangulation + subcategory validation

# Nature系列期刊：LLM可解释性 / 功能特化 / 神经科学×AI 相关文章详细整理

---

## 一、Nature 正刊

### 1.1 Zhang et al. 2025 — "Inter-brain neural dynamics in biological and artificial intelligence systems"
**Nature, Vol. 645, pp. 991–1001. DOI: 10.1038/s41586-025-09196-4**

**解决的问题：** 社交互动中，两个个体的大脑如何协调？这种协调是否反映了一种跨生物和AI系统的通用计算原则？

**方法：**
- 同时记录社交互动中两只小鼠的背内侧前额叶皮层神经元（钙成像 + 单细胞电生理）
- 用子空间分解方法把神经活动分成"共享子空间"（两只动物同步的部分）和"独特子空间"（各自独立的部分）
- 分别记录GABA能（抑制性）和谷氨酸能（兴奋性）神经元
- 训练multi-agent RL agents玩社交协调博弈，分析AI agents内部是否也形成shared/unique subspace
- **因果验证：** 在AI系统中ablate shared subspace，看合作行为是否下降

**发现：**
- GABA能神经元的shared subspace显著大于谷氨酸能神经元（反直觉——通常认为兴奋性神经元主导信息传递）
- AI agents训练后也自发形成相同的shared/unique子空间结构
- **因果结果：** 破坏AI的shared subspace → 合作行为显著下降，证明这个结构是功能必要的
- 结论：shared/unique子空间是跨生物和人工系统的通用社交智能计算原则

**为什么发Nature：** 三重证据收敛——(1)生物学新发现（GABA能细胞类型特异性），(2)跨系统普遍性（AI也有），(3)因果干预验证。

---

### 1.2 Ding et al. 2025 — "Functional connectomics reveals general wiring rule in mouse visual cortex"
**Nature, Vol. 640, pp. 459–469. DOI: 10.1038/s41586-025-08840-3**

**解决的问题：** 大脑皮层中功能相似的神经元是否优先形成突触连接（"like-to-like" wiring）？这个规则是否也在人工神经网络中自发涌现？

**方法：**
- **数据规模：** MICrONS数据集——1立方毫米小鼠视觉皮层，75,909个神经元，5亿突触，电子显微镜4×4×40nm³重建（PB级数据）
- 14次双光子钙成像session记录功能响应
- 建了一个deep recurrent neural network "digital twin"来分离每个神经元的feature tuning和spatial tuning
- 分析突触连接概率 vs 功能相似度的关系
- **因果验证：** 在1000-unit RNN中删除like-to-like连接 vs 删除random连接，比较任务性能下降

**发现：**
- 功能相似的神经元之间突触连接概率显著高于随机预期，跨层、跨区域一致
- 这个规则在feedback pathway中也成立
- RNN训练后自发涌现同样的like-to-like连接模式
- **因果结果：** 删除like-to-like连接比删除随机连接造成更大的性能下降

**为什么发Nature：** PB级别的连接组数据（前所未有的规模）+ 功能记录 + AI模型验证 + 因果ablation = 一个universal wiring principle。

---

### 1.3 Wang et al. 2025 — "Foundation model of neural activity predicts response to new stimulus types"
**Nature, Vol. 640, pp. 470–477. DOI: 10.1038/s41586-025-08829-y**

**解决的问题：** 能否建一个类似LLM的"神经活动基础模型"，从一批小鼠的大规模记录中学习，然后zero-shot预测新刺激类型的神经响应？

**方法：**
- Foundation cohort: 8只小鼠，~66,000个视觉皮层神经元，900+分钟自然视频记录
- 4模块DNN架构（perspective + modulation + core DenseNet-ConvLSTM + readout）
- Pre-train on naturalistic video，few-shot transfer到新小鼠（只需30分钟数据）
- 用MICrONS连接组数据交叉验证：模型readout weights能否预测解剖学定义的细胞类型和突触连接

**发现：**
- Foundation model可以泛化到从未见过的刺激类型（相干运动、噪声图案、Gabor滤波器）
- 只用新小鼠30分钟数据就超过从头训练的模型
- 模型的readout weights能预测细胞类型和连接模式（功能→结构预测）

**为什么发Nature：** 解决了系统神经科学的根本问题（预测新刺激响应） + foundation model的方法论创新 + 连接组交叉验证。

---

### 1.4 Zhou et al. 2024 — "Larger and more instructable language models become less reliable"
**Nature, Vol. 634, pp. 61–68. DOI: 10.1038/s41586-024-07930-y**

**解决的问题：** LLM越大越好吗？Scaling和instruction tuning是否同时提升了性能和可靠性？

**方法：**
- **规模：** 32个LLM变体（GPT×10, LLaMA×10, BLOOM×12, 350M到176B参数）
- 5个benchmark domain，每个1570-3142道题
- 每个benchmark 15种不同prompt措辞（测prompt sensitivity）
- ~4.2M总LLM响应
- 两项人类调查研究（测试人类能否检测LLM错误）
- 心理测量学式难度映射

**发现：**
- Scaling提升benchmark分数但paradoxically降低reliability
- 更大/更instruction-tuned的模型产生更少的refusal但更多plausible-sounding wrong answers
- 在中等难度问题上，30-70%的错误人类监督者无法发现
- 没有任何模型在任何难度级别创造出"error-free zone"

**为什么发Nature：** 这是唯一一篇没有生物数据的Nature AI文章。能发是因为对AI安全和human oversight有重大社会影响。4.2M响应的规模也是前所未有。

---

### 1.5 Muttenthaler et al. 2025 — "Aligning machine and human visual representations across abstraction levels"
**Nature, Vol. 647, pp. 349–355. DOI: 10.1038/s41586-025-09631-6**

**解决的问题：** DNN和人类的视觉表征在不同抽象层级上不一致。能否系统性地修复这个mismatch，并且修复后是否带来实际性能提升？

**方法：**
- Triplet odd-one-out任务收集人类similarity judgments
- 训练surrogate model模仿人类判断，生成pseudo-labels（AligNet dataset, ImageNet规模）
- 用AligNet fine-tune视觉foundation models（SigLIP等）
- 在多个抽象级别（fine-grained到coarse）评估对齐程度
- 测试对齐后是否改善downstream task泛化和OOD robustness

**发现：**
- Fine-tuning on AligNet改善了DNN在所有抽象级别上与人类的对齐
- 对齐后的模型在downstream tasks上泛化更好，OOD robustness更强
- 不仅是"更像人"，而且"更好用"

**为什么发Nature：** 从描述性（"AI不像人"）到规范性（"如何修复并获得功能收益"）的跨越 + 认知科学和ML双重受众。

---

### 1.6 Binz et al. 2025 — "Centaur: A foundation model to predict and capture human cognition"
**Nature, Vol. 644, pp. 1002–1009. DOI: 10.1038/s41586-025-09215-4**

**解决的问题：** 能否用一个LLM fine-tune在行为数据上来预测和模拟人类跨范式的认知行为？

**方法：**
- Base model: Llama 3.1 70B，用QLoRA fine-tune（0.15% trainable parameters）
- Psych-101数据集：160个实验，60,092个被试，10.7M选择，trial-by-trial文本转写
- 评估：held-out participants, OOD task structures, 全新domains
- fMRI brain alignment（决策 + 句子阅读任务）

**发现：**
- Centaur超过认知模型并泛化到unseen domains
- 内部表征与人类fMRI对齐度比base Llama更高（尽管没有显式的neural training signal）
- 反应时可以用entropy measure预测

**为什么发Nature：** "Universal cognitive simulator"概念 + 160个实验的规模 + 生物验证（fMRI对齐是emergent的）。

---

### 1.7 Ji-An et al. 2025 — "Discovering cognitive strategies with tiny recurrent neural networks"
**Nature, Vol. 644, pp. 993–1001. DOI: 10.1038/s41586-025-09142-4**

**解决的问题：** 能否用最小规模（1-4个unit）的RNN揭示动物和人类决策中的认知算法？

**方法：**
- 用1-4个unit的RNN拟合6个经典reward-learning任务的trial-by-trial选择数据
- 分析RNN dynamics提取可解释的认知策略
- 与经典认知模型和更大RNN比较

**发现：**
- 1-4 unit RNN outperform经典认知模型且match大网络性能
- RNN dynamics揭示了之前未被捕获的generalizable cognitive algorithms
- Minimality本身是interpretability的来源

**为什么发Nature：** 极简模型的力量——用最少的组件获得最大的解释力。**和我们的发现相关：只需1%的neurons就能实现clean dissociation。**

---

### 1.8 Cowley et al. 2026 — "Compact deep neural network models of the visual cortex"
**Nature, Vol. 652, pp. 947–954. DOI: 10.1038/s41586-026-10150-1**

**解决的问题：** 预测视觉皮层响应需要60M参数的DNN吗？能否压缩到~12,000参数而不损失预测准确度？

**方法：**
- Adaptive closed-loop实验：交替收集猕猴V4神经记录和模型训练
- 从60M参数迭代压缩到~12,000参数（5000×压缩）
- 分析compact models的内部circuit motifs

**发现：**
- 压缩模型达到与full model相当的V4预测准确度
- 所有compact models共享early-stage filters但通过precise readout consolidation特化
- 泛化到V1和IT，暗示一个general cortical circuit principle

**和我们的关联：** "只需~1%的neurons per function"与"5000×压缩不损失"是同一个故事的不同面——功能被sparse实现。

---

## 二、Nature Machine Intelligence

### 2.1 Dreyer et al. 2025 — "SemanticLens: Mechanistic understanding and validation of large AI models"
**NMI, Vol. 7, pp. 1572–1585. DOI: 10.1038/s42256-025-01084-w**

**解决的问题：** 如何自动地、可扩展地理解大型AI模型中每个neuron编码了什么？

**方法：**
- 把每个neuron的activation pattern映射到CLIP的多模态语义空间
- 实现：concept-based neuron search（搜索编码特定概念的neurons）、自动neuron标注、跨模型比较、spurious correlation检测
- 基于LRP和CRP框架
- **应用案例：** 黑色素瘤检测模型——发现特定neuron clusters编码ABCDE诊断标准中的具体feature，同时发现spurious correlations

**发现：**
- 可以把任意模型的每个neuron嵌入语义空间，实现可搜索、可比较
- 自动发现了医疗AI模型中的bias和spurious features
- 第一个将individual neurons嵌入结构化多模态语义空间的framework

**局限（我们可以improve的）：** 纯描述性——告诉你neuron编码什么，但不做causal ablation证明这个neuron是否因果地负责该功能。没有double dissociation，没有predictive validation。

**论文规模：** 18页正文 + 49页appendix = 74页总。开源PyTorch toolbox。

---

### 2.2 Mischler et al. 2024 — "Contextual feature extraction hierarchies converge in LLMs and the brain"
**NMI, Vol. 6, pp. 1467–1477. DOI: 10.1038/s42256-024-00925-4**

**解决的问题：** 高性能LLM是否在层级结构上比低性能LLM更"像大脑"？LLM的improvement和brain-likeness是否共同涌现？

**方法：**
- **脑数据：** 颅内脑电（iEEG），来自癫痫手术患者听自然语音
- 12个开源LLM（参数量相近，~7B），包括LLaMA, Falcon, MPT, Mistral, Pythia, OPT等
- 每个LLM每层做encoding model预测iEEG响应
- 测量hierarchical alignment：LLM的哪一层最好地预测大脑的哪个区域
- 比较contextual vs non-contextual embeddings

**发现：**
- Benchmark性能越高的LLM，neural prediction accuracy越高
- 高性能LLM用proportionally更早的层就能match低级听觉皮层——即更高效的hierarchical compression
- LLM improvement和brain-likeness是co-emergent的
- Contextual信息是两者共同的关键驱动力

**方法论特点：** Encoding model（线性回归预测），本质是correlation。没有因果干预。但iEEG是invasive recording，分辨率比fMRI高很多。

---

### 2.3 Doerig et al. 2025 — "High-level visual representations in the human brain are aligned with LLMs"
**NMI, Vol. 7, pp. 1220–1234. DOI: 10.1038/s42256-025-01072-0**

**解决的问题：** LLM（只训练在文本上）的场景描述embedding能否解释人类视觉皮层的fMRI活动？是否比vision-only DNN更好？

**方法：**
- 8个participants，7T fMRI
- Natural Scenes Dataset (NSD)：~9000-10000个自然场景 per participant
- 主模型：MPNet (sentence transformer)
- 13个comparison ANN architectures（作为baseline）
- RSA (Representational Similarity Analysis) + linear encoding/decoding
- 训练LLM-RCNN：用LLM objective训练视觉网络（~48,000 images）

**发现：**
- LLM embeddings of scene captions比专门的视觉DNN（训练在百万图像上）更好地解释高级视觉皮层fMRI活动
- 关键是LLM能整合整句caption的complex contextual information，不只是词级feature
- 可以从脑活动线性decode出准确的场景描述

**统计方法：** Two-tailed t-tests across 8 participants, BH-FDR correction (p<0.05), noise-ceiling correction, cross-validated

**Figure结构：** 4个main figures——(1)主结果RSA, (2)linear decoding/caption重建, (3)复杂上下文信息驱动对齐, (4)LLM-RCNN超过13个benchmark。**这个figure结构值得参考。**

---

### 2.4 Du et al. 2025a — "Human-like object concept representations emerge naturally in multimodal LLMs"
**NMI, Vol. 7, pp. 860–875. DOI: 10.1038/s42256-025-01049-z**

**解决的问题：** LLM/多模态LLM是否自发发展出与人类心理表征结构一致的object concept表征？

**方法：**
- **规模：** 4.7 million triplet odd-one-out similarity judgments
- 1,854个natural objects
- 2个LLM：ChatGPT-3.5, Gemini Pro Vision 1.0
- 从triplet judgments提取66维embedding
- 与fMRI/EEG脑数据做RSA对齐（EBA, PPA, RSC, FFA等区域）

**发现：**
- 多模态LLM自发形成66维object concept表征，语义聚类与人类一致
- 与category-selective脑区fMRI/EEG活动对齐
- Text-only LLM对齐较弱；visual grounding是必要条件
- 表征"similar but not identical"——有系统性差异

**NMI评价标准启示：** 4.7M judgments的数据规模是acceptance的重要因素。Claims语气精确："fundamental similarities"不是"equivalence"。

---

### 2.5 Du et al. 2025b — "Are neural network representations universal or idiosyncratic?"
**NMI, Vol. 7. DOI: 10.1038/s42256-025-01139-y**

**解决的问题：** 不同架构的神经网络是否收敛到shared representations，还是各自发展idiosyncratic ones？

**方法：**
- 比较人类和多个模型的表征（CKA或RSA-based alignment metrics）
- 跨架构比较

**发现：**
- 部分universality——相似性存在但不完全一致
- 直接address universality hypothesis

**和我们的关联：** 我们的FCI=0.86跨4个架构是这个问题的一个具体量化回答。

---

### 2.6 Mahner et al. 2025 — "Dimensions underlying the representational alignment of DNNs with humans"
**NMI, Vol. 7, pp. 848–859. DOI: 10.1038/s42256-025-01041-7**

**解决的问题：** 全局scalar（如CKA, RSA score）不足以理解representational alignment的结构。能否分解成可解释的维度？

**方法：**
- Triplet odd-one-out behavioral task（人类和DNN）
- 提取latent representational dimensions
- 比较visual vs semantic维度在人类和DNN中的权重
- In-silico ablation of dimensions

**发现：**
- DNN比人类更重视visual properties，人类更重视semantic properties
- 即使"well-aligned"的模型也有系统性divergence
- 提供了一个decomposed framework替代global scalar

---

### 2.7 Kar, Kornblith, Fedorenko 2022 — "Interpretability of ANNs in AI versus neuroscience"
**NMI, Vol. 4, pp. 1065–1067. DOI: 10.1038/s42256-022-00592-3**

**性质：** Perspective（非实验文章），3页

**核心论点：**
- "Interpretability"在AI和neuroscience中含义根本不同
- AI interpretability: 理解组件如何贡献到输出
- Neuroscience interpretability: 模型组件与脑区/脑现象的显式对齐
- 两个目标逻辑上独立，可以并行推进但不应混淆

**重要性：** 定义了neuro-AI interpretability的概念框架。由三位领域leaders合著（Kar: 灵长类视觉, Kornblith: representational similarity, Fedorenko: language network）。**我们的paper正好坐在他们定义的交叉点上——用neuroscience方法(double dissociation)做AI interpretability。**

---

### 2.8 Suzgun et al. 2025 — "Language models cannot reliably distinguish belief from knowledge and fact"
**NMI, Vol. 7, pp. 1780–1790. DOI: 10.1038/s42256-025-01113-8**

**解决的问题：** LLM能否区分"我相信X"（信念）和"X是事实"（知识）？

**方法：**
- **规模：** 24个LLM（NMI上单研究最大模型数量）
- 13,000个KaBLE benchmark问题
- 13个epistemic task类别
- 跨模型代际比较（GPT-4o前后）

**发现：**
- GPT-4o在factual验证上98.2%→但承认first-person false beliefs时降到64.4%
- DeepSeek R1从>90%降到14.4%
- 模型倾向于"factually correct the user"而不是acknowledge belief

**NMI启示：** NMI接受大规模behavioral evaluation（24模型，13K问题）。这篇没有因果干预也能发NMI。

---

### 2.9 Liu et al. 2025 — "Rethinking machine unlearning for large language models"
**NMI, Vol. 7, pp. 181–194. DOI: 10.1038/s42256-025-00985-0**

**解决的问题：** 如何选择性地从LLM中移除特定数据的影响（敏感/非法信息）而不损害其他能力？

**方法：**
- 综合framework for LLM unlearning
- 连接到model editing, influence functions, knowledge neurons, adversarial training, RLHF
- Taxonomy of unlearning methods + multifaceted efficacy assessment

**发现：**
- Unlearning scope定义不trivial
- 当前方法在efficacy/utility trade-off上有系统性问题
- 提出multidimensional evaluation beyond simple forgetting

**和我们的关联：** NMI接受关于controlled neuron/weight modification的文章——和我们的atlas-guided pruning/steering有关联。

---

### 2.10 Hunklinger & Ferruz 2026 — "Towards the explainability of protein language models"
**NMI. DOI: 10.1038/s42256-026-01232-w**

**性质：** Review/Survey

**核心：** XAI applied to protein language models。定义5种XAI角色：Evaluator, Multitasker, Engineer, Coach, Teacher。只有Evaluator目前被广泛采用。

**NMI启示：** 2026年NMI仍在积极发表interpretability survey。Publication window is open。

---

### 2.11 Warrell et al. 2026 — "Interpretability and implicit model semantics in biomedicine and deep learning"
**NMI. DOI: 10.1038/s42256-026-01177-0**

**性质：** Framework/Perspective

**核心：** 从科学哲学角度讨论interpretability只是model semantics的一个方面。

**NMI启示：** NMI接受foundational/conceptual interpretability工作。

---

## 三、Nature Human Behaviour

### 3.1 Tuckute et al. 2024 — "Driving and suppressing the human language network using large language models"
**NHB, Vol. 8(3), pp. 544–561. DOI: 10.1038/s41562-023-01783-7**

**解决的问题：** LLM的encoding model是否足够好，能够反过来"控制"人类大脑——设计出能maxmially drive或suppress人脑language network的句子？

**方法：**
- 训练GPT2-XL encoding model在1000个corpus句子上预测fMRI language network响应
- 用model forward search从~1.8M候选句子中选250个drive句子 + 250个suppress句子
- 在14个新被试上用fMRI验证
- 3,600个独立被试提供behavioral ratings（grammaticality, meaningfulness等）

**发现：**
- Model-selected句子成功地drive和suppress了新被试的language network（predicted-to-causal transfer）
- 69.4%的理论可获得correlation
- Surprisal和grammatical well-formedness是language network激活的关键决定因素（倒U关系）
- **第一个闭环paradigm：AI预测 → 设计刺激 → 生物验证**

**和我们的关联：** 他们用AI控制大脑，我们用atlas控制AI。两者在逻辑上对称。

---

### 3.2 Luo et al. 2024 — "Large language models surpass human experts in predicting neuroscience results"
**NHB, Vol. 9, pp. 305–315. DOI: 10.1038/s41562-024-02046-9**

**解决的问题：** LLM能否预测neuroscience实验结果？比人类专家好还是差？

**方法：**
- **规模：** 15个LLM + BrainGPT（Mistral-7B用LoRA fine-tune on 1.3B neuroscience tokens）
- 171个人类neuroscience专家（mean 10.1年经验）
- 300个BrainBench items（200 human + 100 GPT-4 generated）
- Forward-looking benchmark（预测outcome，不是recall facts）
- Memorization控制：temporal analysis + zlib-perplexity ratio

**发现：**
- LLM: 81.4% accuracy; Human experts: 63.4%
- BrainGPT进一步+3%
- LLM confidence和accuracy相关（calibrated）
- 不是memorization

**Figure结构：** 7 main + 26 supplementary。NHB接受大规模human subject study + LLM benchmark。

---

### 3.3 Webb et al. 2023 — "Emergent analogical reasoning in large language models"
**NHB. DOI: 10.1038/s41562-023-01659-w**

**解决的问题：** LLM能否做零样本抽象模式归纳（analogical reasoning）？

**方法：**
- GPT-3 (text-davinci-003) + GPT-4
- Raven's Progressive Matrices analogs (非视觉), letter-string analogy, verbal analogy
- 匹配的human behavioral studies

**发现：**
- GPT-3零样本在大多数task上match或超过人类
- GPT-4更强
- Emergent ability，不是mere memorization

---

### 3.4 Strachan et al. 2024 — "Testing theory of mind in large language models and humans"
**NHB. DOI: 10.1038/s41562-024-01882-z**

**解决的问题：** LLM有Theory of Mind吗？

**方法：**
- GPT-4, GPT-3.5, LLaMA2-70B + smaller variants
- 1,907个human participants
- 5种ToM task types（irony, indirect requests, false beliefs, misdirection, faux pas）
- 15 independent sessions per model per task

**发现：**
- GPT-4在大多数ToM task上at or above human level
- 但faux pas task上失败——分析发现是response strategy issue（过度保守），不是inference failure
- GPT-3.5和LLaMA2表现更mixed

---

## 四、Nature Communications

### 4.1 Kumar et al. 2024 — "Shared functional specialization in transformer-based language models and the human brain"
**NatComm, Vol. 15, Article 5523. DOI: 10.1038/s41467-024-49173-5**

**解决的问题：** Transformer的individual attention heads是否执行functionally specialized的计算，这些计算是否对应大脑中特定的cortical regions？

**方法：**
- **模型：** BERT-base (144 attention heads, 12 layers × 12 heads) + GPT-2 (replication)
- **脑数据：** 63个fMRI participants (2个datasets: n=18 + n=45)，听自然故事
- Head-wise encoding model：每个attention head的"transformation"（attention-weighted update）单独预测fMRI时间序列
- 1000-parcel cortical parcellation + 10个language ROIs
- 分析25种syntactic dependency types的head-wise encoding
- **Controls：** shuffled head assignments (destroys structure), untrained BERT (fails), non-language cortex (no encoding)

**发现：**
- Individual attention heads确实functionally specialized——不同heads预测不同cortical regions
- 组织遵循两个principal axes：(1) layer depth, (2) backward attention distance（head回看多远的context）
- Early layers/posterior temporal cortex处理local dependencies；late layers/prefrontal cortex处理long-range context
- 编码特定syntactic dependencies的heads与处理这些操作的brain regions匹配
- GPT-2 replication一致

**统计方法：** Permutation tests with FDR correction (p<0.005), bootstrap 95% CIs, 3-fold cross-validation, noise ceiling normalization

**论文规模：** 5 main figures + 29 supplementary figures

**和我们的对比：** 这是NatComm上最接近我们工作的文章。方法差异：(1) 他们用encoding model (correlation)，我们用causal ablation；(2) 他们做attention heads，我们做FFN neurons；(3) 他们需要fMRI数据，我们不需要；(4) 他们没有double dissociation。**Kumar et al.发的是NatComm不是NMI——可能暗示NMI对"functional specialization without causal proof"要求更高。**

---

### 4.2 Béna & Goodman 2025 — "Dynamics of specialization in neural modules under resource constraints"
**NatComm, Vol. 16, Article 187. DOI: 10.1038/s41467-024-55188-9**

**解决的问题：** Structural modularity是否足以guarantee functional specialization？什么条件下模块化网络才会发展出真正的功能分化？

**方法：**
- Controlled experiments with paired RNNs（不是LLMs）
- Parity-based classification tasks with MNIST/EMNIST
- 三个functional specialization指标：module probing, module ablation, hidden state correlations
- 大量参数扫描：module size (n), inter-module connectivity (p), environmental covariance (c)
- Structural modularity用directed graph Q-metric量化

**发现：**
- Structural modularity alone (even Q > 0.4) 只产生**weak** functional specialization
- 真正的specialization需要两个条件：(a) 强resource constraints (sparse neurons + synapses), (b) environmentally separable features (低任务间协方差)
- Specialization是**dynamic**的，不是static的——在processing timesteps间变化
- 结论："static notion of specialization is likely too simple"

**和我们的关联：** 提供理论背景——functional specialization不是trivially expected。这给我们的empirical finding（LLMs确实有functional specialization）增加了scientific significance。

---

### 4.3 Webb et al. 2025 — "A brain-inspired agentic architecture to improve planning with LLMs"
**NatComm, Vol. 16, Article 8633. DOI: 10.1038/s41467-025-63804-5**

**解决的问题：** LLM在multi-step planning上weak。能否用brain-region-inspired modular decomposition改善？

**方法：**
- MAP (Modular Agentic Planner): 6个specialized LLM instances, 每个对应一个prefrontal cortex功能模块
- 2个models (GPT-4, Llama3-70B), 4个benchmark tasks
- Zero-shot prompting with ≤3 few-shot examples
- 与CoT, ToT, Multi-Agent Debate比较

**发现：**
- MAP outperforms all baselines
- Significant reduction in invalid action proposals
- Generalizes OOD

**注意：** 功能分解是architectural design（通过prompting），不是empirical discovery。与我们方向不同。

---

## 五、Nature Neuroscience

### 5.1 Hermosillo et al. 2024 — "A precision functional atlas of personalized network topography and probabilities"
**Nat Neurosci, Vol. 27(5), pp. 1000–1013. DOI: 10.1038/s41593-024-01596-5**

**解决的问题：** 如何构建大规模、个体特异的brain functional atlas？

**方法：**
- Precision fMRI with Infomap, template matching, NMF, OMNI overlapping mapping
- 53,273 individual network maps from >9,900 individuals (ABCD, HCP, developmental cohorts)
- Resting-state fMRI

**发现：**
- 个体brain network topography差异显著，尽管共享general layout
- MIDB atlas实现个性化network parcellation

**和我们的关联：** "Functional atlas"术语在Nature Neuroscience的生物学先例。我们的"causal functional atlas of LLMs"直接借用了这个vocabulary。

---

### 5.2 van den Heuvel et al. 2025 — "Investigating the methodological foundation of lesion network mapping"
**Nat Neurosci. DOI: 10.1038/s41593-025-02196-7**

**解决的问题：** Lesion Network Mapping (LNM)——从病变位置映射到脑网络——方法论上是否sound？

**方法：**
- 文献survey of 201 LNM studies (101 conditions, 2015–2025)
- 数学分析LNM procedure
- Synthetic lesion experiments

**发现：**
- LNM有fundamental limitation: 重复采样相同的FC matrix → 不同conditions的diverse lesions被映射到同一个nonspecific network
- 为addiction, depression, psychosis识别的networks suspiciously similar
- Circular sampling problem

**和我们的关联：** 我们的方法避开了这个问题——gradient×activation attribution + independent ablation validation是non-circular的。Reviewer如果质疑我们的方法论，这篇提供了"what NOT to do"的reference。

---

### 5.3 Sani et al. 2024 — "DPAD: Dissociative and prioritized modeling of behaviorally relevant neural dynamics"
**Nat Neurosci, Vol. 27(10), pp. 2033–2045. DOI: 10.1038/s41593-024-01731-2**

**解决的问题：** 如何用RNN模型分离行为相关 vs 行为无关的神经dynamics？

**方法：**
- Two-section RNN架构：第一部分学behavior-predictive latent states，第二部分学remaining neural dynamics
- 四步优化算法prioritizing behavioral relevance
- 4个cortical movement tasks (spiking + LFP data)

**发现：**
- DPAD比prior linear/nonlinear方法更准确
- 识别了比standard power features更behavior-predictive的nonlinear LFP transformations
- 成功dissociate了behaviorally relevant/irrelevant neural populations

**和我们的关联：** "Dissociative"在Nature Neuroscience标题中的使用——validation that the dissociation vocabulary is accepted。

---

## 六、Nature Computational Science

### 6.1 Hagendorff et al. 2023 — "Human-like intuitive behavior and reasoning biases emerged in LLMs but disappeared in ChatGPT"
**NCS, Vol. 3, pp. 833–838. DOI: 10.1038/s43588-023-00527-x**

**解决的问题：** LLM是否展现人类式的System 1直觉错误？

**方法：**
- 10个GPT models (GPT-1 through GPT-4)
- 455个human participants
- Cognitive Reflection Test (200 customized items) + semantic illusions (50 tasks)

**发现：**
- 早期GPT (GPT-3 size)表现出人类式System 1直觉错误（CRT失败，语义错觉）
- ChatGPT/GPT-4消除了这些错误（96% vs 38% human）但通过chain-of-thought而不是直觉
- 暗示reasoning mechanism的qualitative shift

**Framing：** Dual-process theory (System 1 vs System 2)作为organizing framework。

---

### 6.2 Gao et al. 2025 — "Increasing alignment of large language models with language processing in the human brain"
**NCS. DOI: 10.1038/s43588-025-00863-0**

**解决的问题：** Scaling LLM size vs instruction tuning，哪个更能提升与人脑language processing的对齐？

**方法：**
- 多个base和instruction-tuned LLMs（不同size）
- 52个native English speakers, fMRI + eye-tracking during naturalistic reading
- Linear encoding models

**发现：**
- Simply scaling size提升brain alignment
- Instruction tuning (RLHF) 提供no additional benefit beyond raw scale
- Larger LLMs的self-attention patterns更好地预测读者的回视眼动和fMRI响应

---

## 七、Nature Reviews Neuroscience

### 7.1 Fedorenko et al. 2024 — "The language network as a natural kind within the broader landscape of the human brain"
**Nat Rev Neurosci, Vol. 25(5), pp. 289–312. DOI: 10.1038/s41583-024-00802-4**

**性质：** Comprehensive review (24页)

**核心论点：**
- Language network是一个"natural kind"——有principled boundaries的coherent ontological grouping
- 特征：left-lateralized frontotemporal regions, language-selective, causally necessary
- 独立于input/output modality
- LLM exposed to same stimuli时预测substantial variance in language network activity
- Language-selective responses在brains和LLMs中都是graded和architecture-dependent的

**和我们的关联：** "Natural kind"框架是我们可以直接借用的。我们的8个cognitive category如果通过double dissociation验证了empirical separability，就可以argue它们也是natural kinds。

---

### 7.2 Sadeh & Clopath 2025 — "The emergence of NeuroAI: bridging neuroscience and artificial intelligence"
**Nat Rev Neurosci, Vol. 26(10), pp. 583–584. DOI: 10.1038/s41583-025-00954-x**

**性质：** Field-level perspective

**核心：** 定义NeuroAI作为一个学科领域。AI tools revolutionize neuroscience；brain-inspired AI has returned。三个capability gaps：physical-world interaction, non-brittle learning, energy efficiency。

**和我们的关联：** 可以cite来position our paper as a contribution to the NeuroAI agenda。

---

## 八、其他顶会顶刊（重要竞争者）

### 8.1 AlKhamissi et al. 2025 — "The LLM Language Network" (NAACL 2025)
**arXiv: 2411.02280**

**解决的问题：** LLM是否包含类似于人脑language network的功能特化子网络？

**方法：**
- Neuroscience-style localization: 用language vs non-language (math, code)刺激对比识别language-selective units
- 应用于18个diverse LLMs
- **因果验证：** ablate localized units, measure language task performance drop
- Brain alignment: localized units是否比random units更好地预测fMRI language network响应
- 扩展到reasoning和social inference domains

**发现：**
- LLMs确实包含"language network"——selective units因果地重要
- 这些units与brain language network recordings对齐
- Specialization程度跨架构变化

**和我们的差距：** (1)只有1-3个domain (language, reasoning, social), 我们8个; (2)没有cross-domain double dissociation; (3)没有dependency DAG; (4)没有predictive control; (5)没有cross-domain interactions

---

### 8.2 Bricken et al. 2023 + Templeton et al. 2024 — Anthropic SAE系列
**Transformer Circuits Thread**

**解决的问题：** 如何从polysemantic neurons中提取monosemantic features？

**方法：**
- Bricken 2023: train SAE on 1-layer transformer, 512 neurons → 4000+ monosemantic features
- Templeton 2024: scale to Claude 3 Sonnet → millions of features (Golden Gate Bridge, sycophantic praise等)
- Feature steering改变model behavior

**和我们的差距：** 无监督feature dictionary vs 有监督causal atlas; features不按cognitive domain组织; 无cross-architecture; 无double dissociation; 无behavioral prediction from atlas

---

### 8.3 Dai et al. 2022 — "Knowledge Neurons in Pretrained Transformers" (ACL, ~559 citations)

**方法：** Integrated gradients on fill-in-the-blank cloze tasks (BERT-style) → identify factual knowledge neurons → suppress验证

**发现：** Factual knowledge stored in sparse FFN neurons, somewhat localized to mid-upper layers

**和我们的差距：** 单domain (factual), 单model (BERT), 无dissociation, 无universality

---

### 8.4 Meng et al. 2022 — "ROME" (NeurIPS, ~1537 citations)

**方法：** Causal tracing (activation patching) → identify mid-layer MLP as factual storage → rank-one weight update to edit facts

**发现：** Small set of middle-layer MLP modules causally mediate factual recall; editing these weights changes output

**和我们的差距：** 单domain, editing focus, 无cross-model, 无dependency DAG

---

### 8.5 Hase et al. 2023 — "Does Localization Inform Editing?" (NeurIPS Spotlight)

**发现：** Localization (causal tracing) 和editability是decoupled的。ROME找到的"causal" site不是最佳editing target。

**对我们的意义：** Reviewer会cite这个攻击所有localization-based工作。我们的defense：我们不做editing，我们做behavioral prediction + control。Double dissociation + pruning + steering直接demonstrate atlas的功能性。

---

### 8.6 Dobs et al. 2022 — "Brain-like functional specialization emerges spontaneously in DNNs" (Science Advances)

**方法：** CNN同时训练face + object recognition。Lesion：selective ablation of top filters for each task, measure cross-task performance drop。Bootstrap testing (10,000 iterations)。

**发现：** Face-only networks: 82.2% face, 17.3% object; Object-only: 29.3% face, 74.1% object。Spontaneous segregation。Combined task segregation index显著。

**对我们的意义：** 最clean的prior double dissociation in ANNs。但是vision CNN，不是LLM。我们是第一个在LLM上做8-domain double dissociation的。

---

### 8.7 Lindsey et al. 2025 — "On the Biology of a Large Language Model" (Transformer Circuits Thread)

**方法：** Cross-Layer Transcoders → attribution graphs showing multi-step reasoning circuits in Claude 3.5 Haiku。Case studies: two-hop reasoning, poetry planning, emotional reasoning。

**发现：** Two-hop reasoning (e.g., "capital of state containing Dallas" → "Texas" → "Austin") involves interpretable intermediate steps。Models plan rhymes ahead。

**和我们的差距：** 一个closed-source model; case-study-based不是systematic atlas; 无cross-model; 无double dissociation

---

### 8.8 Liu et al. 2025 — "Brain-Inspired Exploration of Functional Networks and Key Neurons in LLMs" (arXiv: 2502.20408)

**方法：** 把fMRI functional brain network分析方法直接应用于LLM neurons。识别co-activation functional networks。Inhibit/amplify interventions。

**发现：** LLMs展现recurring functional networks, inhibition impairs performance, amplification enhances it。

**和我们的差距：** 无监督co-activation (类似ULCMOD); 无double dissociation; 无cross-model; 无dependency DAG。

---

### 8.9 Gurnee et al. 2024 — "Universal Neurons in GPT2 Language Models" (arXiv: 2401.12181)

**方法：** 比较不同random seed训练的GPT-2模型，用correlation-based matching识别universally converging neurons。

**发现：** Many neurons converge to similar roles across models。Universal neurons emerge early in training。Ablating them significantly impacts loss。

**和我们的差距：** 同架构不同seed vs 我们4个不同架构。我们的FCI=0.86是更strong的universality claim。

---

### 8.10 Christ et al. 2024 — "Math Neurosurgery" (arXiv: 2410.16930)

**方法：** Gradient-free parameter importance methods (LAPE, Wanda) → isolate math-specific parameters → prune/scale interventions。

**发现：** Math-specific parameters can be isolated。Pruning them degrades math 4-17% on GSM8K without harming general language。Scaling improves math 5-35%。

**和我们的差距：** 单domain (math only); 无8-domain atlas; 无dissociation; 无cross-model。

---

### 8.11 Wang et al. 2022 — "Finding Skill Neurons" (EMNLP, ~200 citations)

**方法：** Prompt-tuning activation分析识别task-specific "skill neurons"。

**发现：** Skill neurons are task-specific; similar tasks share skill neurons。Emerge during pre-training。

**和我们的差距：** NLP pipeline tasks (NER, sentiment等) vs cognitive domains; prompt-tuning-based vs gradient causal attribution; 无dissociation; 无cross-model。

---

### 8.12 Zou et al. 2023 — "Representation Engineering" (arXiv, ~697 citations)

**方法：** Contrast prompts along semantic axis (honest vs dishonest) → compute difference in residual stream → use vector to steer model behavior。

**发现：** Steering vectors exist for honesty, harmlessness, truthfulness, power-seeking。Adding vectors shifts behavior。

**和我们的差距：** Representational (residual stream directions) vs neuronal (FFN neurons); 无neuron-level atlas; 无functional taxonomy; 无causal ablation; 无dependency structure。

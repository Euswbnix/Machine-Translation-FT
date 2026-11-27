# Waldendorf et al. 2025(作者阅读笔记,2026-11-27)

以下是作者本人的阅读结果,原样保存。文末"Claude 的补充"是讨论意见,不是原文内容。

**论文:Multilingual Data Filtering using Synthetic Data from Large Language Models**

- 作者:Jonas Waldendorf, Barry Haddow, Alexandra Birch, Mateusz Klimaszewski
- 发表:Findings of EMNLP 2025, pp. 9317–9334.
- DOI:10.18653/v1/2025.findings-emnlp.495
- ACL Anthology:https://aclanthology.org/2025.findings-emnlp.495/
- PDF:https://aclanthology.org/2025.findings-emnlp.495.pdf

**重点:Sec 4.4–5, Tables 1、4、13、14, App C、F、G.2**

- [x] **使用的是哪个 COMET-KIWI checkpoint,与我们的 CometKiwi-22 是否相同?**
  - 原文没有明确写出 COMET-KIWI 的 checkpoint ID。
  - Section 4.3 引用 Rei et al. (2022),Footnote 5 说明其 backbone 为 InfoXLM 的修改版本。
  - 这些信息与 `Unbabel/wmt22-cometkiwi-da` 一致,因此很可能使用的是 CometKiwi-22,但无法确认完全相同。
  - 正式记录:**Checkpoint 未明确报告(likely CometKiwi-22, but unconfirmed)。**
  - 注意区分数据筛选指标与下游评估指标。Appendix G.1 的 Footnote 12 明确指出,下游 COMET 评估使用的是 `Unbabel/wmt22-comet-da`,而非 COMET-KIWI。
  - 来源:Section 4.3, p. 9320;Appendix G.1, p. 9333。

- [x] **Appendix C 的近似去重是否也用于 En→De?**
  - 是。近似去重适用于全部实验数据,并非仅用于 En→Ar。
  - 作者最初在 En→Ar 实验中发现,MDFS 筛选出了大量相似句子,导致下游翻译性能下降。
  - 因此,他们使用 BICLEANER 对所有数据进行去重。
  - 原文明确使用了 "deduplicate all data"。
  - 去重使用 `--aggressive_dedup` 参数,以移除 near-duplicates。
  - 去重发生在划分 MDFS Training Data 和 NMT Training Data 之前,因此所有下游筛选方法共享经过近似去重的候选语料。
  - 论文没有报告去重前后的语料规模变化、重复率及具体相似度阈值。
  - 这意味着其 COMET-KIWI 在低保留率下表现不佳的现象发生在已经进行近似去重的语料上,不能简单归因于完全重复或近似重复样本。
  - 来源:Appendix C, p. 9330。

- [x] **Section 5.1 声称保留 1% 时 COMET-KIWI 在两个语言对上均表现更差,是否准确?**

  Table 13:En→De,保留 1%:

  | Method | spBLEU | chrF++ | COMET |
  |---|---:|---:|---:|
  | Random | 32.1 ± 0.2 | 54.7 ± 0.2 | 0.783 ± 0.001 |
  | COMET-KIWI | 32.0 ± 0.3 | 54.8 ± 0.2 | 0.763 ± 0.003 |
  | MDFS-Regression | 34.6 ± 0.1 | 56.5 ± 0.1 | 0.800 ± 0.001 |
  | MDFS-Class | 34.7 ± 0.1 | 56.7 ± 0.1 | 0.801 ± 0.001 |

  COMET-KIWI 相对于 Random:spBLEU −0.1;chrF++ +0.1;COMET −0.020。

  结论:En→De 在 spBLEU 和 chrF++ 上基本持平,其中 chrF++ 甚至略高于 Random。只有 COMET 出现明显下降。原文没有报告针对这些差异的正式显著性检验。

  Table 14:En→Ar,保留 1%:

  | Method | spBLEU | chrF++ | COMET |
  |---|---:|---:|---:|
  | Random | 28.8 ± 0.2 | 49.7 ± 0.1 | 0.803 ± 0.001 |
  | COMET-KIWI | 24.4 ± 0.1 | 45.2 ± 0.0 | 0.749 ± 0.001 |
  | MDFS-Regression | 28.6 ± 0.1 | 48.3 ± 0.0 | 0.789 ± 0.000 |
  | MDFS-Class | 28.5 ± 0.1 | 48.2 ± 0.0 | 0.792 ± 0.001 |

  COMET-KIWI 相对于 Random:spBLEU −4.4;chrF++ −4.5;COMET −0.054。En→Ar 在三个指标上均明显落后于 Random。

  不同保留率下 COMET-KIWI 相对于 Random 的变化:

  | Language | Retention | ΔspBLEU | ΔchrF++ | ΔCOMET |
  |---|---:|---:|---:|---:|
  | En→De | 1% | −0.1 | +0.1 | −0.020 |
  | En→De | 10% | +3.0 | +2.0 | +0.015 |
  | En→De | 25% | +3.0 | +1.9 | +0.017 |
  | En→De | 50% | +2.0 | +1.3 | +0.013 |
  | En→Ar | 1% | −4.4 | −4.5 | −0.054 |
  | En→Ar | 10% | −0.2 | −0.2 | 0.000 |
  | En→Ar | 25% | +0.6 | +0.4 | +0.007 |
  | En→Ar | 50% | +0.8 | +0.8 | +0.009 |

  - Section 5.1 中关于两个语言对均表现更差的说法需要限定比较对象和评价指标。如果比较对象是 MDFS,该说法在 1% 时成立;如果比较对象是 Random,则不能笼统地说 En→De 全面退化。
  - En→De 在 1% 时主要出现 neural metric degradation,而非明显的 lexical metric degradation。
  - En→Ar 在 1% 时才表现出跨指标的一致退化。
  - 表格中的误差为三个随机种子实验的 Standard Error of the Mean(SEM),而非标准差。
  - 来源:Section 5.1, Tables 13–14, p. 9332。

- [x] **CometKiwi-22 的人工打分训练数据是否覆盖 En-Fr 和 En-De?**
  - En→De:是。En→Fr:根据已公开的监督训练数据来源,没有直接覆盖。En→Ar:没有直接覆盖。
  - `Unbabel/wmt22-cometkiwi-da` 的人工监督训练数据主要来自 WMT17–WMT20 Direct Assessments 和 MLQE-PE corpus。
  - Glushkova et al. (2023) Appendix C 列出了 WMT17–WMT20 DA 数据的全部 32 个翻译方向,其中包含 En→De、De→En、De→Fr 和 Fr→De,但不包含 En→Fr 或 Fr→En。
  - Fomicheva et al. (2022) 的 MLQE-PE 包含 En-De、En-Zh、Ru-En、Ro-En、Et-En、En-Ja、Ne-En、Si-En、Ps-En、Km-En 和 En-Cs,也不包含 En-Fr。
  - 因此,虽然 InfoXLM backbone 支持英语、法语和阿拉伯语,但语言支持不代表该翻译方向拥有人工质量评分监督。

  | Language pair | InfoXLM 支持 | 公开 DA 数据直接覆盖 |
  |---|---|---|
  | En→De | 是 | 是 |
  | En→Fr | 是 | 否 |
  | En→Ar | 是 | 否 |

  - Waldendorf et al. 在 Section 3.2 和 Section 5.1 中使用 En→De 有人工 DA 监督、En→Ar 没有人工 DA 监督这一差异,解释 COMET-KIWI 在两个语言对上的表现差异。
  - 对我们的 En→Fr 和 En→De 实验而言,这意味着 QE supervision coverage 可能是一个潜在混杂因素。
  - 但不能据此直接认定 En→Fr 上的 COMET-KIWI 筛选效果一定更差;跨语言泛化能力和语料质量分布也可能影响结果。
  - 上述判断基于公开训练数据说明,并未检查 checkpoint 对应的完整训练样本清单。
  - 额外来源:COMET 官方模型说明 https://github.com/Unbabel/COMET/blob/master/MODELS.md ;CometKiwi-22 https://huggingface.co/Unbabel/wmt22-cometkiwi-da ;Rei et al. (2022) https://aclanthology.org/2022.wmt-1.60/ ;Glushkova et al. (2023), Appendix C https://arxiv.org/html/2305.19144 ;Fomicheva et al. (2022) https://aclanthology.org/2022.lrec-1.530.pdf

- [x] **Appendix F:Random baseline 是否每个 seed 重新抽样,具体怎么抽?**
  - 是。对于包含随机采样的数据筛选方法,作者使用三个不同随机种子生成三个训练数据集。
  - Table 12 报告的随机种子为 42、2025、962。
  - Random baseline 从经过预处理的 NMT Training Data 中随机选取句子,构造相应保留率的训练子集。
  - 因此,不是固定同一个随机子集,仅改变模型初始化种子。
  - 论文没有明确说明:是否使用 uniform sampling without replacement;使用何种随机数生成器或采样函数;不同保留率的数据子集是否相互嵌套;数据抽样种子与模型训练种子是否严格一一对应;相同种子下不同方法是否采用完全配对的实验设置。
  - 在没有原始采样代码的情况下,不应补充这些未报告的实现细节。
  - 这种实验设计使 Random baseline 的 SEM 可能同时包含数据抽样方差和模型训练方差。

  NMT 训练配置:Encoder-decoder Transformer,约 63M 参数,FAIRSEQ,100,000 updates,学习率 5e-4,Adam,Inverse Square Root,warmup 4,000,max tokens 16,000,update frequency 2,dropout 0.3,label smoothing 0.1,每 2,500 updates 存检查点,按验证集 BLEU 选检查点,beam 5,种子 42、2025、962。来源:Section 4.3、Section 5、Appendix F、Table 12。

- [x] **Appendix G.2:筛选组与随机组的差距随训练步数缩小的幅度。**
  - Figure 6 展示的是 En→Ro 医疗领域数据筛选实验,而非 En→De 或 En→Ar 翻译质量筛选实验;该图也不包含 COMET-KIWI 筛选组。
  - 横轴为 2,000–7,000 training updates,纵轴为 chrF++。原文没有提供对应的精确数值表,以下为近似读图结果。

  | Updates | Random chrF++ | 最优筛选组 chrF++ | Gap |
  |---:|---:|---:|---:|
  | 2,000 | ~34.3 | ~42.2 | ~7.9 |
  | 3,000 | ~48.7 | ~54.2 | ~5.5 |
  | 4,000 | ~53.0 | ~57.3 | ~4.3 |
  | 5,000 | ~55.4 | ~58.4 | ~3.0 |
  | 6,000 | ~56.7 | ~59.0 | ~2.3 |
  | 7,000 | ~56.7 | ~59.3 | ~2.6 |

  - 从 2,000 到 7,000 updates,最优筛选组相对于 Random 的领先幅度由约 7.9 chrF++ 缩小至约 2.6 chrF++,减少约 5.3 points,即约 67%。
  - 此处比较的是每个训练步数下表现最好的筛选方法所形成的上包络,不一定始终对应同一个筛选方法。
  - 筛选组在约 5,000 updates 时已经达到随机基线最终能够达到的最高性能(约 58.2 chrF++),但这不代表筛选组与 Random 在第 5,000 步已经完全持平。
  - 结果说明数据筛选的优势部分体现在训练早期的学习效率,随着训练继续,随机组能够追回部分差距;但差距并非严格单调缩小,也不能据此推断在充分训练后必然消失。
  - 不能将该结果直接引用为 COMET-KIWI 筛选与随机筛选之间的收敛证据。
  - 来源:Appendix G.2, Figure 6, p. 9333。

**补充核对:Section 4.4–4.5, Tables 1、4**

- Table 1 衡量的是 MDFS 模型复现 Llama-3.1 人工合成标签的能力,不是与真实人工评分的一致性,也不是下游翻译性能。
- 对于 En→De,MDFS-Regression 在阈值 3、4、5 下的 F1 分别为 0.908、0.777、0.640;En→Ar 相应为 0.920、0.757、0.398。高分阈值下的 F1 下降说明精确识别最高质量样本比区分普通好坏样本更加困难。
- Table 4 测量的是医疗领域筛选语料的 lexical diversity,而非翻译质量筛选语料。作者对保留 100 万句子的语料使用 XLM-R tokenizer,打乱语料后取前 100 万 tokens,统计 unique unigram types 和句长中位数。

  | Method | Arabic unique 1-grams | Romanian unique 1-grams |
  |---|---:|---:|
  | Random | 19,970 | 26,705 |
  | Keyword | 14,621 | 22,251 |
  | MDFS-Ngram (En) | 13,513 | 18,490 |
  | MDFS-Ngram (non-En) | 12,479 | 17,733 |
  | MDFS-Class (En) | 13,900 | 21,225 |
  | MDFS-Class (non-En) | 13,251 | 20,983 |
  | MDFS-Regression (En) | 13,494 | 19,614 |
  | MDFS-Regression (non-En) | 12,938 | 18,702 |

- 相对于 Random,词汇多样性最低的筛选组在 Arabic 和 Romanian 上分别减少约 37.5% 和 33.6% 的 unique unigram types。
- 该结果证明领域筛选可以产生词汇多样性更低的训练子集,但由于筛选组同时具有更高的领域相关性,不能据此建立词汇多样性下降与翻译性能下降之间的因果关系。
- Table 4 没有报告 COMET-KIWI 筛选组的 lexical diversity,也没有分析 bigram、trigram 或 contextual diversity。

**对我们论文的意义**

- Waldendorf et al. (2025) 提供了极端保留率下纯 QE-based filtering 可能不如随机筛选的实验证据。
- 这种现象在已经进行 aggressive near-deduplication 的候选语料中仍然存在。
- 但作者没有直接分析 COMET-KIWI 筛选导致的词汇覆盖损失、上下文多样性损失或重复暴露之间的关系。
- 作者提出的人工 DA 训练语言对覆盖差异,为我们的 En→Fr 与 En→De 实验提供了另一种潜在解释,需要与 lexical diversity hypothesis 区分。
- Table 4 的 unigram diversity 统计可以作为我们测量词汇多样性的相关方法参考,但不能将其描述为 COMET-KIWI 筛选造成词汇坍缩的直接证据。
- Appendix G.2 支持在多个训练 checkpoint 上比较筛选组与随机组,而非仅报告最终模型性能。
- 我们的"重复那一半 vs 窄词表多样组"实验可以进一步检验:当训练预算和词汇覆盖受到控制时,重复暴露与上下文多样性是否能够解释质量筛选造成的性能退化。
- Waldendorf et al. 并未在其论文中进行这一直接控制实验。

**Related Work 英文表述(草稿)**

Waldendorf et al. (2025) found that aggressive quality-based filtering with COMET-KIWI could underperform random selection when retaining only 1% of the available parallel data, despite prior near-duplicate removal. For English–German, the degradation was primarily reflected in COMET, whereas English–Arabic exhibited consistent degradation across spBLEU, chrF++, and COMET. The authors hypothesized that differences in human-annotated QE training coverage could partly explain the language-dependent outcomes. However, they did not directly investigate whether reduced lexical coverage, limited contextual diversity, or repeated exposure contributed to these failures. Our controlled experiments aim to disentangle these factors under comparable training budgets.

---

## Claude 的补充(2026-11-27)

1. **最重要的一条:CometKiwi-22 的人工打分训练数据不覆盖 En-Fr。** 我们的主实验(E0.4–E0.6)恰好就在 En-Fr 上。审稿人可以借 Waldendorf 的解释说:顶部段不如随机,是 CometKiwi-22 在一个没有人工监督的语言对上校准不好,而不是"重复"。这正是计划风险表第 12 条。分两层回应:
   - 对主张一(顶部段内部,重复那一半 vs 多样那一半):两半来自同一个分数段,平均 QE 分实测几乎相同(0.8990 / 0.8993)。"校准不好"要解释的是同一分数下为什么重复的句子伤害更大,它解释不了这一点。
   - 对主张二(顶部段 vs 随机):"校准不好"确实是一个竞争解释。E0.7 用的 En-De 有人工监督;如果效应在 En-De 上也复现,这个解释就弱了很多。所以 E0.7 现在多了一个作用,预注册里应该写明。W1(换一个 QE 模型)也能回应这一条。
2. **随机组要不要每个种子重抽(E0.6 第 12 项)。** 他们每个种子重抽(42、2025、962),表中的 ± 是 3 个种子的 SEM,里面同时有抽样波动和训练波动。我们 E0.5 是一次固定抽样、5 个训练种子共用。草稿推荐沿用固定抽样;两种都说得通,论文里写明即可。和他们的数字比较时要注意:SEM 不能直接和我们的种子标准差比。
3. **学习率一项。** 他们是从头训练(学习率 5e-4、inverse sqrt、10 万步),不是微调,所以对我们的"激进学习率"质疑没有直接参照。
4. **Appendix G.2 和我们的 10k 步。** 你读得对:那是 En→Ro 领域筛选,不含 COMET-KIWI,不能当收敛证据。但它提示了一个审稿问题:差距会不会随训练继续缩小?E0.6 如果在中间检查点(比如每 2,500 步)也解码一次 newstest2014,就能画出差距随步数的曲线。成本只是多几次解码(5090 上几分钟一次),建议作为描述量登记进预注册。
5. **Table 4 的做法和我们一致。** 他们在固定 100 万 token 下数 unique unigram types,和我们"固定 6.5M 法语 token 下数词型"是同一类做法,引用时可以说方法一致。
6. **Related Work 草稿。** 事实部分准确。两处小改:(a) 他们的检查点没写明,写"COMET-KIWI"而不是"CometKiwi-22"(你已经这样写了,保持);(b) 最后一句"disentangle these factors"同上一篇的意见,E0.6 只能分开覆盖和完全重复的源句,建议等结果出来再定措辞。

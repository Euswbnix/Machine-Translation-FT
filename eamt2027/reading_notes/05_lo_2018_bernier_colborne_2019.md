# Lo et al. 2018 + Bernier-Colborne & Lo 2019(作者阅读笔记,2026-11-27)

以下是作者本人的阅读结果,原样保存。文末"Claude 的补充"是讨论意见,不是原文内容。

**论文 1:Accurate semantic textual similarity for cleaning noisy parallel corpora using semantic machine translation evaluation metric: The NRC supervised submissions to the Parallel Corpus Filtering task**

- 作者:Chi-kiu Lo, Michel Simard, Darlene Stewart, Samuel Larkin, Cyril Goutte, Patrick Littell
- 发表:WMT 2018, pp. 908–916.
- ACL Anthology:https://aclanthology.org/W18-6481/ ;PDF:https://aclanthology.org/W18-6481.pdf

**论文 2:NRC Parallel Corpus Filtering System for WMT 2019**

- 作者:Gabriel Bernier-Colborne, Chi-kiu Lo
- 发表:WMT 2019, pp. 252–260. DOI:10.18653/v1/W19-5434
- ACL Anthology:https://aclanthology.org/W19-5434/ ;PDF:https://aclanthology.org/W19-5434.pdf

**补充文献**

- Koehn et al. (2019), Findings of the WMT 2019 Shared Task on Parallel Corpus Filtering for Low-Resource Conditions:https://aclanthology.org/W19-5404.pdf
- Littell et al. (2018), Measuring sentence parallelism using Mahalanobis distances: The NRC unsupervised submissions to the WMT18 Parallel Corpus Filtering shared task:https://aclanthology.org/W18-6480.pdf

---

## 一、Lo et al. 2018

**重点:Sections 2.1、2.4、3.2,Tables 1–3**

- [x] **10M / 100M 词对应多少句对、占候选池多少、是否按英语词计?**

  | Property | Value |
  |---|---:|
  | Original sentence pairs | 104M |
  | Original English tokens | ~1B |
  | Original German tokens | 964M |
  | Sentence pairs after NRC pre-filtering | 28M |
  | Sentence-pair retention after pre-filtering | ~26.9% |
  | 10M-word budget relative to original pool | ~1% |
  | 100M-word budget relative to original pool | ~10% |

  - Section 1 明确报告原始 ParaCrawl 候选池包含 104 million German–English sentence pairs。词数在 punctuation tokenization 之前统计。
  - Section 2.1 的预过滤将候选池从 104M 句对缩减至 28M 句对。
  - 10M 和 100M 均按照英语端词数计算,由 Koehn et al. (2018), Sections 3、5.2 明确确认。
  - 原文没有报告各筛选方法在 10M / 100M 预算下最终保留的具体句对数量。
  - 由于不同方法选出的句子具有不同长度,不能简单地将 1% 和 10% 的词数比例转换成 1.04M 和 10.4M 句对。
  - 预过滤后的 28M 句对包含多少英语 tokens 也没有报告,因此不能计算两个预算相对于预过滤后候选池的准确保留比例。
  - 正式记录:**104M original sentence pairs, reduced to 28M after pre-filtering; 10M and 100M English-word budgets correspond to approximately 1% and 10% of the original English token count. Actual selected sentence-pair counts were not reported.**
  - 来源:Lo et al. (2018), Sections 1、2.1;Koehn et al. (2018), Sections 3、5.2。

- [x] **bicov 的平均 +2.44 / +1.61 BLEU 是什么的平均?**

  Table 2:加入 bicov 前后的 BLEU 差值。

  | Scorer | Model | Budget | Dev baseline | Dev bicov | ΔDev | Test baseline | Test bicov | ΔTest |
  |---|---|---|---:|---:|---:|---:|---:|---:|
  | HMM | SMT | 10M | 19.09 | 20.42 | +1.33 | 23.55 | 25.31 | +1.76 |
  | YiSi-1 | SMT | 10M | 21.56 | 22.19 | +0.63 | 24.68 | 27.41 | +2.73 |
  | HMM | NMT | 10M | 21.16 | 23.17 | +2.01 | 26.59 | 29.08 | +2.49 |
  | YiSi-1 | NMT | 10M | 24.24 | 26.69 | +2.45 | 30.75 | 33.56 | +2.81 |
  | HMM | SMT | 100M | 24.42 | 24.68 | +0.26 | 29.73 | 29.98 | +0.25 |
  | YiSi-1 | SMT | 100M | 24.47 | 24.84 | +0.37 | 30.10 | 30.46 | +0.36 |
  | HMM | NMT | 100M | 31.53 | 31.98 | +0.45 | 39.52 | 39.66 | +0.14 |
  | YiSi-1 | NMT | 100M | 32.49 | 33.20 | +0.71 | 40.27 | 40.98 | +0.71 |

  - 10M SMT 平均增益:(1.33 + 1.76 + 0.63 + 2.73) / 4 = 1.6125 ≈ 1.61 BLEU。
  - 10M NMT 平均增益:(2.01 + 2.49 + 2.45 + 2.81) / 4 = 2.44 BLEU。
  - 即 +1.61 / +2.44 是 HMM 和 YiSi-1 两个打分器,在 newstest2017 和 newstest2018 两个评估集上的四项配对增益的算术平均。不是多个随机种子的平均,不是六个官方测试领域的平均,更不是相对于 Random 的增益。
  - 如果只考虑 YiSi-1,10M NMT 的平均增益为 +2.63 BLEU。
  - 100M 下 SMT 和 NMT 的平均增益分别约为 +0.31 和 +0.50 BLEU。bicov 在较小训练语料预算下的改善更加明显。

  Table 3 的官方评估结果:

  | Model | Budget | NRC-yisi | NRC-yisi-bicov | Difference |
  |---|---|---:|---:|---:|
  | SMT | 10M | 23.52 | 23.89 | +0.37 |
  | SMT | 100M | 26.37 | 26.40 | +0.03 |
  | NMT | 10M | 26.50 | 27.41 | +0.91 |
  | NMT | 100M | 31.76 | 31.88 | +0.12 |

  - 官方评估中的改善明显小于 NRC 内部实验。作者认为这可能与内部 MT 模型和组织方 MT 模型的架构差异有关。
  - Table 3 的评估平均值与 Koehn et al. (2018) 的 Table 4 基本对应,开发集虽然单独展示,但根据各列数值复算,不计入该 average 列。
  - 引用 +2.44 / +1.61 时必须注明是内部实验结果,不能将其描述为 WMT 官方评估增益。
  - 原文未报告多种子方差或正式显著性检验。
  - 来源:Section 3.2、Table 2、Section 4、Table 3,pp. 913–914。

- [x] **bicov 丢掉句子后怎么补足预算?评估方向是什么?**

  Section 2.4 定义的是基于源端 word bigram 的硬过滤方法。设 q(sᵢ,tᵢ) 为句对的原始质量评分,G₂(sᵢ) 为源句的 word bigram types,H 为此前已经覆盖的 bigram 集合。

  1. 按原始质量评分从高到低排列全部候选句对。
  2. 依次检查每个句对的源端 bigrams。
  3. 如果 G₂(sᵢ) \ H 非空,则保留该句对。
  4. 将该句对中的 bigrams 加入 H。
  5. 如果没有任何新的 bigram,则移除该句对。

  - 只要包含一个此前未覆盖的源端 bigram,即满足保留条件。不要求引入新的 unigram,也不要求达到某个新 bigram 比例。
  - 不考虑 bigram 出现频率、目标词频分布或外部参考词表。
  - 冗余过滤发生在质量评分之后,因此优先保留质量评分较高的相似样本。
  - Section 3.2 明确说明作者使用组织方提供的子集提取软件构造 10M 和 100M English-word 训练集。根据这一流程,被过滤句对占用的预算可以由后续符合条件的候选句对补足。
  - 但原文没有报告具体的补足代码、被过滤句对对应的提交分数、边界句子的处理方式,以及候选语料不足时是否存在 fallback。"继续向下扫描直到满足预算"是合理实现,而非原文完整公开的算法细节。
  - 正式记录:**Quality-ranked hard filtering requiring at least one previously unseen source-side word bigram, followed by fixed English-word-budget subset extraction. Exact refill and fallback implementation details were not reported.**

  | Property | NRC internal evaluation | Official WMT18 evaluation |
  |---|---|---|
  | Translation direction | De→En | De→En |
  | Source language for bicov | German | German |
  | SMT | Portage | Moses |
  | NMT | Sockeye 1.18.20 Transformer | Marian v1.5 |
  | Internal dev set | newstest2017 | — |
  | Internal test set | newstest2018 | — |
  | Official test sets | — | Six domains |
  | Metric | BLEU | BLEU |

  - NRC 内部 NMT 明确使用 self-attention Transformer,而官方 WMT18 NMT 使用 Marian v1.5 默认设置,不能将两者视为相同模型。
  - NRC 内部 Transformer 使用默认参数,但最大源端和目标端序列长度均改为 60,gradient clipping 设置为 1。
  - 来源:Sections 1、2.4、3.2;Koehn et al. (2018), Sections 5.2–5.3。

- [x] **bicov 只在 YiSi-1 和 HMM 上测过,还是也在其他打分器上测过?**

  Lo et al. (2018), Table 2:

  | Scorer | Without bicov | With bicov | Paired ablation |
  |---|---|---|---|
  | HMM p(e\|f) | Yes | Yes | Yes |
  | YiSi-1 precision | Yes | Yes | Yes |
  | Linear regression ensemble | No | Yes | No |
  | Random scoring | Yes | No | No |

  - Lo et al. 论文内部只有 HMM 和 YiSi-1 提供完整的有无 bicov 配对消融。
  - regression bicov(L1-regularized linear regression 的特征组合加 bicov)没有报告 without bicov 的版本。
  - Table 1 中的 YiSi-2、perplexity、POS perplexity 和其他特征仅进行了句对质量识别能力评估。

  同年 NRC 的另一篇论文 Littell et al. (2018) 还在无监督打分方法中使用了 bicov:

  | Scorer | Without bicov | With bicov | Paired ablation |
  |---|---|---|---|
  | Unsupervised feature ensemble | NRC-mono | NRC-mono-bicov | Yes |
  | Mahalanobis ratio | Not reported | NRC-seve-bicov | No |

  Littell et al. (2018), Table 4:

  | Model | Budget | NRC-mono | NRC-mono-bicov | Difference |
  |---|---|---:|---:|---:|
  | SMT | 10M | 19.79 | 20.97 | +1.18 |
  | SMT | 100M | 26.01 | 26.23 | +0.22 |
  | NMT | 10M | 20.69 | 23.07 | +2.38 |
  | NMT | 100M | 31.20 | 31.56 | +0.36 |

  - NRC-mono 是多个无监督特征的等权组合;NRC-seve-bicov 以 Mahalanobis ratio 为主要打分特征。补充论文同样观察到小预算下 bicov 的增益更大。
  - 但 Littell et al. 的 Section 2.4 将其 post-filtering 描述为基于 target-language sentence,而 Lo et al. 明确写的是 source-language bigram。不能未经代码核实就断言两个版本使用完全相同的语言端。
  - 结论:bicov 并非仅在 YiSi-1 和 HMM 上得到测试;它也被应用于其他监督和无监督打分器,但并非每个打分器都有完整的配对消融。
  - 来源:Lo et al. (2018), Tables 1–2;Littell et al. (2018), Sections 2.3–2.4、Table 4。

**补充:Lo et al. 2018 对高排名句子冗余的明确观察**

- Section 2.4 明确指出,句对语义相似度评分并不直接衡量训练样本对 MT 的效用。
- 作者观察到排名最前的样本存在明显冗余及有限的词汇覆盖,例如大量正确翻译的日期表达。
- 这一问题发生在 NRC 已经执行额外预过滤之后。预过滤包括:将 email 和 URL 替换为占位符后去重;移除数字匹配不足的句对;去掉数字及标点后进行进一步去重;过滤语言识别不符合要求的句对;过滤任一端超过 150 tokens 的句对。
- 因此,Lo et al. 已经发现普通去重及模板式去重不足以完全解决高质量排名中的词汇覆盖和冗余问题。
- 这是目前已核查文献中非常早期且直接的相关观察,但尚未进行系统性的更早文献检索,不应直接宣称其为历史上首次发现该现象。

---

## 二、Bernier-Colborne & Lo 2019

**重点:Sections 2.3、3.1–3.2,Tables 4–5**

- [x] **候选池大小、Table 5 使用的组织方系统是什么?**

  Bernier-Colborne & Lo (2019) 未报告完整候选池的具体句对及词数统计。根据 Koehn et al. (2019), Table 5:

  | Language pair | Sentence pairs | English words |
  |---|---:|---:|
  | Ne→En | 2,235,512 | 58,537,167 |
  | Si→En | 3,357,018 | 60,999,374 |

  - 原始数据为经过句对去重的 ParaCrawl 爬取结果,组织方未进行其他过滤。
  - NRC 随后执行自己的预过滤(email、URL 和数字掩码后的去重、数字匹配检查、语言识别及长度过滤),并额外移除 Nepali/Sinhala 端中英语、数字和标点占比超过 50% 的句对。NRC 没有报告预过滤后的候选池规模。

  Table 5 使用的是 WMT19 组织方的统一下游评估系统,而不是 NRC 自己的 Portage 系统。根据 Koehn et al. (2019), Section 5.4、Figure 1:SMT 为 Moses(5-gram 语言模型,最大短语长度 5,最大句长 80);NMT 为 fairseq Transformer(FLORES baseline 配置:5 层编码器、5 层解码器,embedding 512,FFN 2048,2 个注意力头,共享 embedding,SentencePiece 词表 5000,dropout 0.4,attention dropout 0.2,ReLU dropout 0.2,label smoothing 0.2,Adam,学习率 1e-3,warmup 4000,max tokens 4000,update frequency 4,最多 100 epochs,beam 5,length normalization 1.2);评估为 case-insensitive SacreBLEU。

  - 官方报告中训练 1M 和 5M 子集分别需要约 3 小时和 13 小时,硬件为单张 GTX 1080 Ti。
  - 总参数量、实际 optimizer updates 和累计训练 token exposure 未报告。
  - 注意区分 NRC 使用的 XLM Transformer 句对质量打分模型与组织方用于下游翻译评估的 fairseq Transformer。
  - 来源:Bernier-Colborne & Lo (2019), Section 4;Koehn et al. (2019), Sections 4.4、5.4、Table 5、Figure 1。

- [x] **1M / 5M 词是否按英语端计?**

  - 是。Koehn et al. (2019), Sections 3、5.3 明确说明预算按照英语端词数计算。

  | Language pair | 1M-word retention | 5M-word retention |
  |---|---:|---:|
  | Ne→En | ~1.71% | ~8.54% |
  | Si→En | ~1.64% | ~8.20% |

  - 上述比例根据官方 Table 5 的精确英语词数计算。官方 Abstract 将两种保留率近似描述为 2% 和 10%。
  - 2019 年仍然使用基于质量分数阈值的固定词数子集提取;对于阈值处分数相同的句对,组织方随机选择其中一部分以满足预算。
  - 来源:Koehn et al. (2019), Sections 3、5.3、Table 5。

- [x] **Table 4 的 Zipporah* 一行是什么设置?**

  Table 4:官方 dev-test 上的 SMT uncased BLEU。

  | Method | Ne→En 1M | Ne→En 5M | Si→En 1M | Si→En 5M |
  |---|---:|---:|---:|---:|
  | Zipporah* | 3.40 | 4.22 | 4.16 | 4.77 |
  | Random | 1.30 | 3.01 | 1.43 | 3.33 |
  | Zipporah | 4.14 | 4.42 | 4.12 | 4.96 |
  | YiSi-2-sup | 3.86 | 3.76 | 4.85 | 4.71 |
  | YiSi-2-unsup | 4.42 | 3.91 | 3.97 | 4.56 |
  | XLM-v2-spc | 4.14 | 4.09 | 4.52 | 4.72 |
  | XLM-v2-spc-mlm | 3.96 | 3.69 | 4.37 | 4.68 |
  | XLM-v3-spc-mlm | 3.89 | 3.91 | 4.12 | 4.66 |
  | Ensemble | 3.94 | 3.95 | 4.89 | 4.85 |

  - Zipporah* 是任务组织方报告的 Zipporah baseline,使用组织方自己的 SMT 系统(Moses)。
  - 不带星号的 Zipporah 是 NRC 获取组织方提供的 Zipporah score lists 后,用其选出的语料训练 NRC 自己的 Portage SMT 系统得到的结果。
  - 两行的主要明确区别是下游 SMT 系统不同,而非两种不同的 Zipporah 筛选算法。星号不代表统计显著性,也不是加入 coverage re-ranking 的标志。
  - 与 NRC 自己的方法进行公平比较时,应优先使用不带星号的 Zipporah。
  - 特别注意:Table 4 的 Random 并不是直接对原始候选池进行纯随机采样。它包含 NRC 的 initial filtering 和 token coverage re-ranking,不能视为未经任何覆盖优化的 uniform random baseline。
  - 来源:Table 4 caption、Section 3.2, pp. 257–258。

- [x] **覆盖重排(不带新二元组的句对打八折)有没有任何消融?**

  Section 2.3 的 coverage re-ranking 是 Lo et al. (2018) 硬过滤方法的软惩罚版本。设 qᵢ 为句对原始质量分数,G₂(sᵢ) 为源端 word bigram 集合,Hᵢ₋₁ 为此前更高评分句对所覆盖的 bigrams:G₂(sᵢ) \ Hᵢ₋₁ 非空则 qᵢ′ = qᵢ;为空则 qᵢ′ = 0.8 × qᵢ。

  - 基于原始质量评分降序遍历;只检查源端(Nepali 或 Sinhala);只统计 word bigrams;至少一个新 bigram 即可避免惩罚。
  - 没有新 bigram 的句对不会被直接删除,只将原始分数降低 20%,仍可能进入最终训练子集。
  - 原文没有说明为什么选择 20%,也没有报告其他 discount ratios 的比较;没有明确报告是否多轮迭代重排,或如何处理 score ties。最自然的复现方式是按原始排名单次扫描和降分,然后重新排序。

  **消融核对结果:没有报告覆盖重排对下游 MT 性能的独立消融。**

  - Section 2 明确说明四个 NRC 提交系统共享 initial filtering、sentence pair scoring 和 coverage re-ranking。
  - Table 4、Table 5 都没有提供同一打分器有无 re-ranking 的配对结果。
  - Random baseline 本身也经过 coverage re-ranking,因此不能利用 Random 与 YiSi/XLM 的差异估计覆盖重排的独立贡献。
  - Section 3.1 的 Footnote 10 提到未应用 re-ranking,但这一设置仅用于检查 XLM 原始打分的高排名句子(作者由此发现 Nepali 高排名样本中存在较多圣经文本,推测存在 domain bias),属于定性分析。
  - Section 5 将进一步改进 vocabulary coverage re-ranking 列为未来研究方向。
  - 正式记录:**A fixed 20% score discount was applied to sentence pairs containing no unseen source-side word bigram. No downstream ablation or discount-strength sensitivity analysis was reported.**
  - 来源:Sections 2.3、3.1、3.2、5,Tables 4–5。

**补充核对:Table 5 的官方 NMT 结果**

| Language | System | SMT 1M | SMT 5M | NMT 1M | NMT 5M |
|---|---|---:|---:|---:|---:|
| Ne→En | YiSi-2-sup | 3.55 | 4.07 | 3.06 | 1.34 |
| Ne→En | YiSi-2-unsup | 4.04 | 4.14 | 3.74 | 0.98 |
| Ne→En | XLM-v2-spc | 3.92 | 4.51 | 4.03 | 1.40 |
| Ne→En | Ensemble | 4.10 | 4.30 | 4.58 | 1.10 |
| Si→En | YiSi-2-sup | 3.87 | 4.39 | 4.97 | 1.58 |
| Si→En | YiSi-2-unsup | 3.14 | 4.29 | 2.41 | 0.68 |
| Si→En | XLM-v2-spc | 3.80 | 4.42 | 1.63 | 0.91 |
| Si→En | Ensemble | 4.19 | 4.54 | 4.06 | 1.39 |

- 所有 NRC 系统的 NMT 性能在 5M-word 条件下均低于 1M-word 条件。作者推测原始候选池中适合 NMT 训练的有效平行数据不足 5M words。
- SMT 则普遍在更大预算下取得更好的结果,作者认为这反映了 SMT 对噪声的更强鲁棒性。
- 这一结果不能证明 coverage re-ranking 导致性能下降,因为所有 NRC 系统均使用该步骤,且没有对应的消融实验。
- 作者明确表示由于计算资源限制,在开发阶段没有进行 NMT 测试。因此,20% 降分系数没有经过下游 NMT 的系统性调优。

---

## 三、2018 与 2019 方法的直接比较

| Property | Lo et al. 2018 | Bernier-Colborne & Lo 2019 |
|---|---|---|
| Motivation | Reduce redundancy | Reduce redundancy |
| Initial ranking | Quality score | Quality score |
| Coverage language | Source | Source |
| Feature | Word bigram | Word bigram |
| Novelty threshold | At least one unseen bigram | At least one unseen bigram |
| No novel bigram | Hard rejection | Score × 0.8 |
| External reference vocabulary | No | No |
| Frequency-aware coverage | No | No |
| Dynamic marginal gain optimization | No | Not reported |
| Paired downstream ablation | Yes | No |
| Penalty-strength sensitivity | No | No |
| Main language pairs | De→En | Ne→En、Si→En |
| Budget unit | English words | English words |

- 两种方法的核心差异是 hard filtering 与 soft re-ranking。2018 方法严格排除不能提供新 bigram 的句对;2019 方法允许高质量但不包含新 bigram 的句对继续参与竞争。
- 但 2019 论文没有直接证明软降分优于硬过滤。
- 两种方法都采用二元的 novelty criterion,只关心是否存在至少一个新 bigram,而不区分引入 1 个还是 20 个新 bigrams。
- 两种方法均不能保证 unigram vocabulary coverage、词频分布或语义多样性得到最优控制。

---

## 四、对我们论文的意义及必须考虑的补救基线

这两篇论文与我们的研究最直接相关,因为它们已经明确描述了独立句对质量评分可能产生的高排名冗余和词汇覆盖不足问题。因此,我们不能将以下内容单独作为创新点:

- 首次发现质量评分最高的句子存在冗余。
- 首次发现质量筛选可能降低词汇覆盖。
- 首次提出在质量排序之后引入 n-gram coverage。
- 首次提出对不包含新 n-grams 的样本进行降分。

如果本文提出针对 COMET-KIWI 筛选的 diversity-aware mitigation method,建议至少包含以下基线:

| Baseline | Definition |
|---|---|
| Random | 随机选择固定预算训练子集 |
| QE Top-k | 直接按 COMET-KIWI 分数选择 |
| QE + Exact Dedup | 质量排序结合完全去重 |
| QE + Near Dedup | 质量排序结合近似去重 |
| QE + BiCov-Hard | 不包含新源端 bigram 的句对直接排除 |
| QE + BiCov-Soft | 不包含新源端 bigram 的句对分数乘以 0.8 |
| Proposed Method | 我们提出的补救方法 |

- QE + BiCov-Hard 是 Lo et al. (2018) 的方法迁移到 COMET-KIWI;QE + BiCov-Soft 是 Bernier-Colborne & Lo (2019) 的方法迁移到 COMET-KIWI。两者均应标注为 adapted baselines,而非完全复现原始 WMT 系统。
- 建议保持相同的候选池、预过滤、tokenizer、质量评分、训练预算、optimizer updates 和模型初始化协议。
- 对所有方法同时报告源端和目标端的 unigram coverage、bigram diversity、trigram diversity、重复率、平均 QE 分数和下游翻译性能。
- 若使用可能包含负值的 QE 分数,需要注意直接乘以 0.8 可能反而提高负分样本的排名。应明确分数归一化或非负化方式,并将其报告为复现时的适配决策。
- 对 hard filtering,还应明确无法达到目标预算时的 fallback 策略。原论文未提供这一实现细节。
- 如果只研究失效机制而不提出新筛选算法,bicov 仍然适合作为验证机制是否具有实际干预价值的补充实验,但不必将其视为所有因果分析的必要对照组。

**尚未解决的研究问题**

- Lo et al. (2018) 已经证明 bigram coverage heuristic 可以改善部分质量打分器的下游表现,但没有独立控制词汇覆盖、翻译质量及上下文多样性。
- Bernier-Colborne & Lo (2019) 将硬过滤改为软惩罚,但没有独立消融,因此无法判断该步骤本身的贡献。
- 两篇论文均没有研究在保持词汇覆盖和训练预算相近时,重复暴露与不同上下文组合之间的差异;均未直接分析 lexical diversity 与 contextual diversity 对性能的独立贡献。
- 因此,我们的"重复那一半 vs 窄词表多样组"对照可以进一步检验早期 coverage-based heuristics 背后的作用机制。
- 研究贡献应强调 controlled disentanglement 和机制验证,而非重新发现高排名冗余或简单提出覆盖重排。

**Related Work 英文表述(草稿)**

Lo et al. (2018) observed that independently scoring parallel sentence pairs by semantic similarity could result in substantial redundancy and limited vocabulary coverage among highly ranked examples. To mitigate this problem, they introduced a hard bigram coverage filter that retained sentence pairs only when they introduced previously unseen source-side word bigrams. Their internal experiments showed average improvements of 2.44 BLEU for NMT and 1.61 BLEU for SMT under a 10-million-English-word budget, averaged across two scoring methods and two evaluation sets. Bernier-Colborne and Lo (2019) subsequently introduced a softer variant that discounted the scores of sentence pairs containing no unseen source-side bigrams by 20%, although no isolated downstream ablation of this re-ranking step was reported. These studies establish important precedents for coverage-aware quality filtering. However, they do not disentangle lexical coverage, repeated exposure, contextual diversity, and translation quality under controlled training budgets. Our experiments investigate these mechanisms directly, while adapting the earlier hard and soft bigram coverage heuristics as mitigation baselines.

---

## Claude 的补充(2026-11-27)

1. **bicov 对我们这种"重复"很可能几乎不起作用,值得先用 CPU 量一下。** bicov 的条件是"只要有一个没见过的源端 bigram 就保留"。我们的重复那一半,重复的是句子之间共享的 8 词片段;这样的句子几乎总还带着别的新 bigram,所以 bicov 大概删不掉多少(推断,没量过)。建议作为一项描述性的 CPU 检查:在我们的顶部段上按 QE 分从高到低跑一遍 bicov,报告被删(或被降分)句子的比例,以及剩下的集合"重复程度"降了多少。如果几乎不变,这本身就是一个有用的结果:NRC 的补救针对的是"完全没有新内容的句子"(比如日期行),和我们量到的重复不是同一种。不需要 GPU。
2. **在我们的分数段里,"软降分"和"硬删除"实际上是一回事。** 顶部段的 CometKiwi 分数挤在 0.896–0.914,乘以 0.8 变成约 0.72,直接掉出这一段。所以照搬 20% 的折扣,BiCov-Soft 和 BiCov-Hard 选出的集合会相同。要让软版本有意义,折扣得按段内分数的跨度重新定(这是适配决策,要写明);你提到的负分问题在我们这里不存在(分数都是正的)。
3. **他们的"日期行"和我们的数字检查正好对上。** NRC 预过滤已经做了"去掉数字和标点后去重",顶部仍然冗余。我们 §3.2 第 2 项(10 月 19 日):遮掉数字、日期、UN 文件编号后,重复那一半仍有 95% 的句子含重复 8 词片段,两半差距保留 88%。两边都说明:这种冗余不是数字模板能解释的。
4. **基线表和计划里的 W4 一致。** W4 的设计就是"NRC 重排、去重、同尺寸随机"三类对照。它在计划里是可选的补救实验;做不做是范围决定(之前我建议至少做 W1、W2)。你最后一条判断我同意:只研究机制、不提补救方法的话,bicov 不是必需的对照;一旦论文里提任何补救方法,这张表里的基线就都是必需的。第 1 条的 CPU 检查不受这个决定影响,两种情况下都值得做。
5. **2019 年那张表的 Random 不是纯随机,你抓得对。** 它也经过预过滤和覆盖重排。所以这篇不能用来说"打分系统胜过随机多少"。
6. **Related Work 草稿。** 事实准确。建议:(a) 在 +2.44 / +1.61 后面补上官方评测的 +0.91 / +0.37,免得被说只挑内部数字;(b) 写明打分器是 YiSi-1(先机器翻译再比语义相似度)和 HMM,不是无参考 QE 模型,2019 年的 YiSi-2 是词向量相似度;(c) 最后一句"while adapting ... as mitigation baselines"只有做了 W4 才能写;"disentangle"一句同前几篇,等 E0.6 结果再定。
7. **Littell 2018 的 bicov 写的是目标端、Lo 2018 写的是源端**,这一点你已经标出。引用时分别按各自原文写,不要合并成"NRC 的 bicov 是源端的"。

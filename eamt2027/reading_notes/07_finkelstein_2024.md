# Finkelstein et al. 2024(作者阅读笔记,2026-11-27)

以下是作者本人的阅读结果,原样保存。文末"Claude 的补充"是讨论意见,不是原文内容。

**论文:Introducing the NewsPaLM MBR and QE Dataset: LLM-Generated High-Quality Parallel Data Outperforms Traditional Web-Crawled Data**

- 作者:Mara Finkelstein, David Vilar, Markus Freitag(Google)
- 发表:WMT 2024, pp. 1355–1372. DOI:10.18653/v1/2024.wmt-1.126
- ACL Anthology:https://aclanthology.org/2024.wmt-1.126/ ;PDF:https://aclanthology.org/2024.wmt-1.126.pdf

**重点:Section 4.1, Tables 4、6、7、16, Figure 2**

- [x] **QE 筛选集的确切大小是多少?**

  Section 4.1 明确说明,使用 BLEURT-QE 从 WMT23 sentence-level training dataset 中选择质量最高的句对,使筛选后的数据规模与对应翻译方向的 NewsPaLM MBR sentence-level dataset 完全一致。根据 Table 2:

  | Property | En→De | De→En |
  |---|---:|---:|
  | Original WMT23 pool | ~296M | ~296M |
  | QE-filtered subset | 998,435 | 1,022,344 |
  | Approximate retention | 0.3373% | 0.3454% |
  | Approximate pruning rate | 99.6627% | 99.6546% |

  - 998,435 和 1,022,344 是根据 Section 4.1 的等规模说明及 Table 2 确定的,并非 Table 4 直接列出的数字。
  - 原文报告的候选池大小为 296 million sentence-level examples,没有提供精确到个位的原始句对数量。原文 Section 4.1 自己也将保留比例近似描述为 0.3%。
  - 上述保留率按照句对数量计算,而不是 source tokens、target tokens 或累计训练 tokens。
  - 论文没有报告筛选后真实平行语料的平均句长或总 token 数,因此无法确认其与 NewsPaLM 在实际训练 token 数上是否匹配。
  - 完整 WMT23 baseline 还可以使用从具有文档边界的语料构造的 blob-level examples,而 BLEURT-QE 筛选组明确为 sentence-level only。

  正式记录:**The BLEURT-QE-filtered subsets contained 998,435 En→De and 1,022,344 De→En sentence pairs, corresponding to approximately 0.34% of the original WMT23 parallel corpus.**

  来源:Sections 2.5、3.1.1、4.1,Tables 2、4。

- [x] **候选池来源是否只包括 Europarl、News Commentary、Rapid?是否包含 UN 等模板化语料?**

  - Section 3.1.1 使用的是完整 WMT23 德英平行训练语料,而不是仅使用 Europarl、News Commentary 和 Rapid。
  - 论文点名这三个语料库,是因为其中约 300 万句具有文档边界,可以用于构造 blob-level training examples。这三个语料库并不是 296M 候选池的完整来源清单。
  - Finkelstein et al. 没有逐项报告完整候选池的语料构成、各来源的句对数量,以及 BLEURT-QE 筛选前后的来源比例。

  根据 WMT23 官方训练数据页面,German–English 方向提供的平行数据包括:Europarl v10、ParaCrawl v9、Common Crawl、News Commentary v18.1、Wiki Titles v3、WikiMatrix、Tilde MODEL、Back-translated News、TED Talks、OPUS;**不包括 UN Parallel Corpus V1.0**。

  - WMT23 官方 UN Parallel Corpus V1.0 提供给 Russian–English 和 Chinese–English,而不是 German–English。因此,没有证据表明 Finkelstein et al. 的德英候选池包含 UN Parallel Corpus V1.0。
  - 但 Europarl、Rapid、Tilde MODEL 和 OPUS 等来源仍然可能包含制度化表达、固定句式或重复模板。
  - 官方允许使用这些数据,不代表 Finkelstein et al. 一定使用了每一个来源。论文没有给出完整的数据构建清单。
  - 也没有报告 BLEURT-QE top-ranked samples 是否集中于某些模板化领域。因此,不能将筛选后的性能退化直接归因于 UN 语料、模板句过度集中或重复样本。
  - 原文也没有明确报告候选池的全局 exact deduplication、near-deduplication 算法或相似度阈值。

  正式记录:**The candidate pool was the WMT23 German–English parallel training corpus. Europarl, News Commentary, and Rapid were specifically identified as the document-boundary-bearing subset. The complete corpus composition and post-filtering source distribution were not reported. UN Parallel Corpus V1.0 was not officially provided for German–English in WMT23.**

  来源:Finkelstein et al. (2024), Section 3.1.1, p. 1358;WMT23 官方训练数据 https://www2.statmt.org/wmt23/translation-task.html

- [x] **BLEURT-QE checkpoint、是否为纯 top-k、随机种子、训练步数和学习率。**

  **QE checkpoint:**

  - Section 4.1 明确使用 BLEURT-QE,并引用 Peter et al. (2023)。没有报告具体 checkpoint ID、checkpoint 路径、模型参数量或版本编号。
  - 根据 Peter et al. (2023), Section 3,其 BLEURT-QE 使用修改后的 mT5 encoder-decoder 架构,以源句和候选译文作为输入,预测翻译质量分数;使用 WMT 2015–2020 Direct Assessment 数据训练。
  - Peter et al. 还说明,其改进版本由 Juraska et al. (2023) 提交并公开,但不能据此认定 Finkelstein et al. 使用的是该公开版本。
  - 不能将 BLEURT-QE 等同于标准 reference-based BLEURT-20,也不能将其等同于 COMET-KIWI。

  | Application | Metric |
  |---|---|
  | Filtering existing WMT23 parallel data | BLEURT-QE |
  | NewsPaLM sentence-level MBR decoding | BLEURT |
  | NewsPaLM blob-level QE reranking | MetricX-QE |
  | Downstream checkpoint selection | BLEURT |
  | Downstream evaluation | BLEURT, MetricX, COMET20, COMET22 |

  上述五种用途应严格区分。尤其是 Table 4 中的 BLEURT-QE-filtered WMT23 与 QE blob-level NewsPaLM,虽然都涉及 QE,但并不是同一种数据筛选方法。

  **是否为纯 top-k:**

  - Section 4.1 明确表示选择 WMT23 候选池中质量最高的样本,并将数量限制为与 NewsPaLM sentence-level dataset 完全一致。这在方法描述层面对应固定样本数量的 QE top-k selection。
  - 没有报告在 BLEURT-QE 排序后加入 vocabulary coverage、bigram coverage、clustering、diversity-aware reranking 或其他补救措施。
  - 但原文没有提供完整筛选代码、分数阈值、score ties 处理方式以及额外预过滤细节。可以将其归类为 reported quality-only top-k selection,但不能断言所有未报告的预处理均不存在。
  - Section 4.3.2 的 cluster-based sampling 用于 NewsPaLM 合成数据,不是 WMT23 BLEURT-QE 筛选流程的一部分。

  **下游 NMT 模型配置:**

  | Parameter | Value |
  |---|---|
  | Architecture | Encoder-decoder Transformer |
  | Parameters | 602M |
  | Framework | Pax |
  | Encoder / decoder layers | 8 / 8 |
  | Model dimension | 1024 |
  | FFN hidden dimension | 8192 |
  | Attention heads | 16 |
  | Vocabulary size | 32K bilingual subword units |
  | Vocabulary training data | Full WMT23 training corpus |
  | Label smoothing | None |
  | Checkpoint selection | Highest development-set BLEURT |
  | Learning rate, optimizer, LR scheduler, warmup, batch size | Not reported |
  | Training epochs, optimizer updates, training-token budget | Not reported |
  | Random seeds, number of independent runs | Not reported |

  - BLEURT-QE 筛选实验使用的是从头训练的 602M encoder-decoder Transformer,而不是对 PaLM-2 Bison 进行微调。
  - 32K bilingual vocabulary 在完整 WMT23 数据上训练,而不是分别根据各筛选子集重新训练。这一设置控制了 tokenizer vocabulary 的差异,但没有保证筛选子集实际覆盖相同数量的 subword types。
  - 全文未报告训练随机种子、学习率、optimizer updates 或累计训练 token 数;没有报告不同训练组是否使用完全相同的计算预算,也没有报告多种子方差或显著性检验。
  - 因此,不能排除极端数据筛选组的性能退化部分来自训练不足、重复暴露差异或学习率调度问题。
  - Peter et al. (2023) 的训练超参数不能直接作为 Finkelstein et al. (2024) 的训练超参数。

  正式记录:**BLEURT-QE checkpoint: not reported. Selection: highest-scoring examples under a fixed sentence-pair budget. Downstream model: 602M Transformer trained from scratch. Random seeds, learning rate, optimizer updates, and cumulative token exposure were not reported.**

  来源:Finkelstein et al. (2024), Sections 3.2–3.3、4.1;Peter et al. (2023), Section 3:https://aclanthology.org/2023.wmt-1.50/

- [x] **Table 16 中"微调对数据量不敏感"的实验使用什么学习率和训练步数?**

  **原文没有报告学习率和训练步数。**

  Section 4.3.1、Figure 3 和 Appendix B 的 Table 16 比较完整 NewsPaLM MBR sentence-level dataset 与其随机抽取的 25% 子集(En→De 998,435 对约 249,609;De→En 1,022,344 对约 255,586,后者为推算值)。

  - 子集使用随机抽样,而非 BLEURT-QE top-k 筛选。
  - PT 表示从头训练 602M Transformer;FT 表示对预训练后的 encoder-decoder Transformer 进行微调。根据 Section 4.2 的默认设置,FT 从完整 WMT23 数据训练得到的 checkpoint 初始化。该实验不是 PaLM-2 Bison 的微调实验。
  - 论文没有说明完整数据组和 25% 数据组是否使用相同的 optimizer updates、epochs、learning rate 或累计训练 token 数。

  **Table 16 完整结果:**

  | Direction | Stage | Dataset | BLEURT ↑ | MetricX ↓ | COMET20 ↑ | COMET22 ↑ |
  |---|---|---|---:|---:|---:|---:|
  | En→De | PT | Full | 56.93 | 6.39 | 21.86 | 72.05 |
  | En→De | PT | 25% | 43.73 | 11.02 | −25.72 | 59.75 |
  | En→De | FT | Full | 70.65 | 2.30 | 55.38 | 82.69 |
  | En→De | FT | 25% | 70.00 | 2.55 | 53.91 | 82.11 |
  | De→En | PT | Full | 67.21 | 4.97 | 44.12 | 80.55 |
  | De→En | PT | 25% | 60.43 | 7.41 | 22.89 | 75.91 |
  | De→En | FT | Full | 73.56 | 2.91 | 61.47 | 84.57 |
  | De→En | FT | 25% | 73.15 | 2.96 | 59.35 | 84.33 |

  从完整数据减少至 25% 后的性能变化:

  | Direction | Stage | ΔBLEURT | ΔMetricX | ΔCOMET20 | ΔCOMET22 |
  |---|---|---:|---:|---:|---:|
  | En→De | PT | −13.20 | +4.63 | −47.58 | −12.30 |
  | En→De | FT | −0.65 | +0.25 | −1.47 | −0.58 |
  | De→En | PT | −6.78 | +2.44 | −21.23 | −4.64 |
  | De→En | FT | −0.41 | +0.05 | −2.12 | −0.24 |

  - 对 En→De,从头训练的 COMET22 下降 12.30 points,而微调仅下降 0.58 points;De→En 分别为 4.64 和 0.24。
  - 作者据此认为,在其高质量 NewsPaLM 数据和预训练模型条件下,微调性能对数据规模的敏感性低于从头训练。
  - 但该实验仅比较 100% 和 25% 两种规模,没有测试更低保留率,也没有提供完整 scaling curve。
  - 由于训练步数和学习率未报告,不能判断微调组是否通过增加重复暴露弥补了数据规模减少。
  - 不能将该结论推广为所有 NMT 或 LLM fine-tuning 对数据规模均不敏感;也不能引用 Table 16 证明极端 QE filtering 后的性能差距会随训练步数增加而消失。

  正式记录:**Reducing the NewsPaLM MBR dataset to 25% caused substantially larger performance degradation during training from scratch than during fine-tuning. Learning rates, optimizer updates, epochs, and token exposure were not reported, preventing a compute-controlled interpretation.**

  来源:Section 4.3.1、Figure 3、Appendix B、Table 16, p. 1372。

---

**补充核对:Table 4 中 BLEURT-QE 筛选的性能退化**(WMT23 test set,602M Transformer 从头训练)

| Direction | Training dataset | MetricX ↓ | COMET22 ↑ |
|---|---|---:|---:|
| En→De | WMT23 Full | 4.20 | 78.79 |
| En→De | WMT23 BLEURT-QE filtered | 16.69 | 43.18 |
| En→De | NewsPaLM Greedy sentence + blob | 2.60 | 81.67 |
| En→De | NewsPaLM MBR sentence | 6.39 | 72.05 |
| En→De | NewsPaLM QE blob | 2.82 | 80.62 |
| En→De | NewsPaLM MBR + QE | 2.99 | 79.68 |
| De→En | WMT23 Full | 5.55 | 82.41 |
| De→En | WMT23 BLEURT-QE filtered | 14.80 | 57.27 |
| De→En | NewsPaLM Greedy sentence + blob | 3.47 | 83.30 |
| De→En | NewsPaLM MBR sentence | 4.97 | 80.55 |
| De→En | NewsPaLM QE blob | 4.01 | 82.33 |
| De→En | NewsPaLM MBR + QE | 3.95 | 82.02 |

Appendix B, Table 9 的四项指标:

| Direction | Dataset | BLEURT ↑ | MetricX ↓ | COMET20 ↑ | COMET22 ↑ |
|---|---|---:|---:|---:|---:|
| En→De | WMT23 Full | 64.11 | 4.20 | 42.52 | 78.79 |
| En→De | WMT23 QE-filtered | 29.33 | 16.69 | −1.14 | 43.18 |
| En→De | NewsPaLM MBR sentence | 56.93 | 6.39 | 21.86 | 72.05 |
| De→En | WMT23 Full | 69.96 | 5.55 | 51.52 | 82.41 |
| De→En | WMT23 QE-filtered | 44.80 | 14.80 | −64.27 | 57.27 |
| De→En | NewsPaLM MBR sentence | 67.21 | 4.97 | 44.12 | 80.55 |

BLEURT-QE 筛选组相对于完整 WMT23 数据:En→De ΔBLEURT −34.78、ΔMetricX +12.49、ΔCOMET20 −43.66、ΔCOMET22 −35.61;De→En −25.16、+9.25、−115.79、−25.14。

相对于等句对规模的 NewsPaLM MBR sentence-level 数据:En→De −27.60、+10.30、−23.00、−28.87;De→En −22.41、+9.83、−108.39、−23.28。

- 极端 BLEURT-QE 筛选后的性能下降覆盖两个翻译方向和全部四项报告指标。
- 但论文没有提供从相同 WMT23 候选池随机抽取约 100 万句对的对照组。因此,不能直接声称 BLEURT-QE top-k 不如同规模 Random。
- 与完整候选池的比较同时改变了训练数据规模、数据组成及可能的训练计算预算。与等规模 NewsPaLM MBR 的比较则同时改变了数据来源、翻译生成方式、语言分布、翻译风格和潜在训练 token 数。两种比较均不能独立识别 QE ranking 本身造成的性能损失。
- 作者指出,其约 0.3% 的保留率远低于此前证明 QE filtering 有效的研究所使用的保留率,因此认为两类结果并不矛盾。
- 作者没有针对真实平行语料的词汇覆盖、n-gram diversity 或高排名冗余提出直接的实验验证。

**补充核对:Figure 2 的含义**

- Figure 2 比较 NewsPaLM MBR sentence-level 与 NewsPaLM QE blob-level 数据从头训练后的性能(En→De,WMT23)。横轴为 source segment length,而非训练步数。图中没有 WMT23 BLEURT-QE filtered 组。
- Sentence-level 模型随着测试序列长度增加而明显退化,blob-level 模型相对稳定。作者据此认为,训练阶段接触多句子数据对 paragraph-level translation 的泛化能力非常重要。
- 这里的 contextual exposure 指跨句子、段落级上下文,不等于我们研究的固定词表下不同句子或 n-gram 组合的 contextual diversity。Figure 2 不能直接作为 QE top-k 导致 lexical coverage collapse 的证据。
- 此外,两个训练集不仅在上下文长度上不同,也分别使用 MBR decoding 和 QE reranking,因此不是完全隔离上下文长度的控制实验。

Table 3 的平均序列长度(Moses tokenizer):En→De sentence-level 37.5 / 39.8(源 / 目标),blob-level 364.5 / 339.8;De→En sentence-level 77.3 / 88.3,blob-level 288.4 / 323.4。

- Blob-level 数据的单个样本明显更长。虽然 QE blob-level 的样本数量低于 100 万,但其训练文本量可能显著高于 MBR sentence-level 数据。不能将相同数量的 sentence examples 和 blob examples 视为相同训练 token budget。

**补充核对:Table 6 的微调结果**(602M encoder-decoder Transformer 的微调,不包含 PaLM-2 Bison)

| Direction | Fine-tuning dataset | MetricX ↓ | COMET22 ↑ |
|---|---|---:|---:|
| En→De | Greedy sentence + blob | 2.59 | 81.49 |
| En→De | MBR sentence | 2.30 | 82.69 |
| En→De | QE blob | 2.45 | 81.83 |
| En→De | MBR sentence + QE blob | 2.26 | 82.52 |
| De→En | Greedy sentence + blob | 3.12 | 84.14 |
| De→En | MBR sentence | 2.91 | 84.57 |
| De→En | QE blob | 2.99 | 84.27 |
| De→En | MBR sentence + QE blob | 2.82 | 84.53 |

- 与从头训练不同,MBR sentence-level 数据在微调阶段取得更高的 COMET22。作者推测预训练阶段获得的长上下文处理能力在微调期间得到保留;这属于作者的机制假设,而非直接验证的因果结论。
- Table 6 没有报告 BLEURT-QE-filtered WMT23 的微调结果。因此,不能声称相同的 BLEURT-QE 筛选策略在从头训练时失效、但在微调时恢复有效。

**补充核对:Table 7 的 cluster-based data selection 消融**

- NewsPaLM 使用聚类后的均匀抽样,避免原始 Newscrawl 分布中大型 cluster 占据过多样本。Section 4.3.2 对两种 25% 子集进行比较:按 cluster 均匀抽样;近似按原始 Newscrawl cluster 分布抽样(作者无法精确重建原始分布)。这不是 WMT23 BLEURT-QE 数据筛选的消融。

| Direction | Stage | Sampling | MetricX ↓ | COMET22 ↑ |
|---|---|---|---:|---:|
| En→De | PT | Uniform over clusters | 11.02 | 59.75 |
| En→De | PT | Original distribution | 11.60 | 59.09 |
| En→De | FT | Uniform over clusters | 2.55 | 82.11 |
| En→De | FT | Original distribution | 2.55 | 81.88 |
| De→En | PT | Uniform over clusters | 7.41 | 75.91 |
| De→En | PT | Original distribution | 7.58 | 76.55 |
| De→En | FT | Uniform over clusters | 2.96 | 84.33 |
| De→En | FT | Original distribution | 3.09 | 83.92 |

均衡采样相对于原始分布的 ΔCOMET22:En→De PT +0.66、FT +0.23;De→En PT −0.64、FT +0.41。

- 聚类均衡采样在多数条件下带来小幅改善,但效果并不一致(De→En 从头训练时 MetricX 改善而 COMET22 下降)。没有多随机种子或显著性检验。
- 该实验构成 cluster-based diversity-aware data selection 的已有研究先例。如果我们的补救方法采用语义聚类均衡采样,应考虑将其作为相关基线,但需要明确这是对原方法的迁移。

**与其他 QE filtering 研究的比较**

| Property | Chimoto et al. 2024 | Tan et al. 2024 | Finkelstein et al. 2024 | Waldendorf et al. 2025 |
|---|---|---|---|---|
| Downstream model | Transformer | LLaMA2-7B | 602M Transformer | ~63M Transformer |
| Training regime | From scratch | Fine-tuning | From scratch | From scratch |
| QE scorer | COMET-QE | COMETKiwi | BLEURT-QE | COMET-KIWI |
| Exact filtering checkpoint | Not reported | Not confirmed | Not reported | Not reported |
| Language pairs | En-De, En-Sw | En-Zh | En-De, De-En | En-De, En-Ar |
| Main aggressive QE condition | 10% | ~0.135% | ~0.34% | 1% |
| Same-pool Random comparison | Yes | Ambiguous | No | Yes |
| Equal-size alternative comparison | Yes | Yes | Synthetic MBR data | Yes |
| Training updates | Not reported | Not reported | Not reported | 100K |
| QE-specific lexical diversity analysis | No | No | No | No |
| Controlled contextual diversity experiment | No | No | No | No |

**对我们论文的意义**

- Finkelstein et al. (2024) 提供了 COMET 系列之外的 QE scorer 在极端质量筛选条件下产生较差下游翻译结果的证据。
- 使用 BLEURT-QE 将约 296M 句对缩减至约 1M 后,En→De 和 De→En 的 COMET22 分别比完整语料训练下降 35.61 和 25.14 points。
- 这支持 aggressive QE-based data selection 的潜在问题并非 COMET-KIWI 所独有,但不能证明不同 QE scorers 的失效机制相同。
- 该论文缺少同一候选池、相同句对规模的 Random baseline,因此无法确定极端筛选造成的退化有多少来自数据规模减少,有多少来自质量排名诱导的数据分布变化。
- 原文没有报告 BLEURT-QE 筛选后的词汇覆盖、n-gram diversity、句对重复率、词频分布或来源集中程度。
- 没有报告学习率、训练步数、种子及累计 token exposure,使训练不足和重复暴露成为无法排除的混杂因素。
- Table 16 表明在该实验条件下,从头训练比微调更加依赖数据规模,但没有严格控制训练计算预算。
- Table 7 已涉及语义聚类均衡采样,因此不能将 cluster-based diversity-aware selection 单独作为全新的方法。
- Figure 2 所研究的段落级上下文暴露,与我们的窄词表下上下文组合多样性不是同一个变量。
- 我们的实验应通过同源候选池、相同训练预算、统一 tokenizer,以及对 lexical coverage、contextual diversity 和 repeated exposure 的独立控制,区分这些潜在机制。

**Related Work 英文表述(草稿)**

Finkelstein et al. (2024) reported substantial degradation when training a 602M encoder-decoder Transformer on approximately one million sentence pairs selected from a 296-million-pair WMT23 corpus using BLEURT-QE. Relative to training on the full corpus, COMET22 decreased by 35.61 and 25.14 points for English-to-German and German-to-English, respectively. However, the study did not include an equally sized random subset from the same candidate pool, nor did it report the lexical coverage, contextual diversity, or cumulative training-token exposure of the selected subsets. Consequently, the observed degradation cannot be attributed specifically to quality-based ranking rather than reduced corpus size or other confounding factors. Our controlled experiments seek to isolate these mechanisms by comparing quality-selected and randomly selected data under matched training budgets, while independently examining lexical coverage, contextual diversity, and repeated exposure.

---

## Claude 的补充(2026-11-27)

1. **这篇里最干净的比较是"QE 筛选集对 NewsPaLM MBR 句级数据"。** 两边都只有单句、句对数相同(约 100 万),所以"只用单句训练、却在段落级测试集上评"的问题两边都有,解释不了它们之间约 23–29 分 COMET22 的差距。但数据来源和译文生成方式不同,仍不是随机对照。引用 −35.6 / −25.1(相对全量)时,要同时写出这两条限定:全量可以用段落级样本,筛选集只有单句。
2. **你查到德英池子里没有 UN 语料,这一条有用。** BLEURT-QE 最顶端在一个没有 UN 的池子里照样崩,说明"QE 最顶端有问题"不依赖 UN 文本;但这篇没看顶端选出了什么,所以只能说现象不限于我们的池子,不能说机制相同。
3. **Table 16 和我们的设置同构,审稿人可能两头用。** 他们的微调也是"先在整个池子上训练、再在子集上微调",结论是微调对数据量不敏感(只掉 0.2–0.6 分)。审稿人可以说"微调本来就不敏感,你的 2.63 BLEU 差距可疑,多半是学习率";我们也可以反过来说"在一个对数据量不敏感的设置里,两半仍差 2.63 BLEU(t = 33),说明差的是内容"。他们没报告学习率和步数,所以帮不了我们回答学习率质疑,还是要靠自己的学习率对照。
4. **Table 7 是"多样性采样"的先例,但效应很小(±0.2 到 ±0.7 分)。** 如果做 W4,聚类均衡采样可以算一个基线;同时它也说明"加多样性就能明显变好"并不是已有的结论。
5. **对 W1(换一个 QE 模型)的启发。** BLEURT-QE 没有可确认的公开检查点;公开的非 COMET 家族 QE 模型里,MetricX 的 QE 版本是一个候选。就目前读过的论文看,还没有人在"最顶端切片、同池同尺寸随机对照"下测过非 COMET 的 QE 模型(Peter et al. 的 BleurtQE 对随机是在保留 50% 时)。这是 W1 能填的空。
6. **次要指标的选择得到印证。** 他们报 MetricX 和 COMET22,用 BLEURT 选检查点。我们计划的 chrF 加 MetricX-24 与此一致;COMET-22 和选数据的 CometKiwi-22 同族,报的话要标明。
7. **请再核一个数。** Table 3 里 De→En 句级平均长度 77.3 / 88.3,是 En→De 句级 37.5 / 39.8 的两倍。两个方向的句级数据来源相同的话不该差这么多,请看一下是不是抄错了行,或者原文单位不同。
8. **Related Work 草稿。** 事实准确。建议补"trained from scratch"和"the filtered subset was sentence-level only, while the test set is paragraph-level";最后一句里"comparing quality-selected and randomly selected data under matched training budgets"我们确实做了(E0.5),可以保留;"independently examining ... contextual diversity"同前几篇,等结果再定。

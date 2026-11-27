# Chimoto et al. 2024(作者阅读笔记,2026-11-27)

以下是作者本人的阅读结果,原样保存。文末"Claude 的补充"是讨论意见,不是原文内容。

**论文:Critical Learning Periods: Leveraging Early Training Dynamics for Efficient Data Pruning**

- 作者:Everlyn Asiko Chimoto, Jay Gala, Orevaoghene Ahia, Julia Kreutzer, Bruce A. Bassett, Sara Hooker
- 发表:Findings of ACL 2024, pp. 9407–9426.
- DOI:10.18653/v1/2024.findings-acl.560
- ACL Anthology:https://aclanthology.org/2024.findings-acl.560/
- PDF:https://aclanthology.org/2024.findings-acl.560.pdf

**重点:Sec 4–5, Tables 4–6, App A–B**

- [x] **Tables 4–6 的加粗代表什么?"±"是 bootstrap 区间还是种子波动?**
  - 已直接检查 PDF 中的三个表格及其 captions。
  - 加粗代表 CAT-DIFF 相对于对应方法具有统计显著的优势(p < 0.05),而不是该单元格取得最高分。
  - Appendix B 明确说明使用 paired bootstrap resampling,重采样次数为 1,000,置信水平为 95%。
  - 零假设为 CAT-DIFF 与被比较方法的平均翻译得分相同。
  - BLEU 和 chrF++ 的"±"根据 Appendix B 所述的评估流程及 SacreBLEU 默认配置,应解释为 bootstrap 95% confidence interval 的半宽,而非多个训练种子的标准差或 SEM。论文没有单独给出"±"的符号定义。
  - COMET 部分没有报告"±",仅报告得分及显著性标记。
  - 需要注意,Figure 4 的误差条采用不同定义:German 和 French 的误差条表示五次不同数据选择随机种子实验的最低及最高 BLEU。
  - Tables 4–6 的 bootstrap 不确定性与 Figure 4 的 seed variability 不应混淆。
  - 表格中的显著性检验针对 CAT-DIFF 与其他方法,并非 COMET-QE 与 Random 的直接比较。
  - 来源:Appendix B, pp. 9423–9426;Figure 4, p. 9415。SacreBLEU 文档:https://github.com/mjpost/sacreBLEU

- [x] **COMET-QE 保留的是高分一端吗?**
  - 原文没有明确描述 COMET-QE 的排序方向、具体筛选阈值或采样算法。
  - Section 4.4 仅介绍 COMET-QE 作为基于 XLM-RoBERTa 的翻译质量估计方法。
  - Section 5.2 在解释 Swahili 的结果时指出,选择质量最好的句对能够确保高质量翻译样本被保留。这间接支持其使用高分端筛选。
  - 因此,保留 COMET-QE 高分端是合理推断,但不是原文明确定义的实现细节。
  - 正式记录:**Presumably top-scoring COMET-QE sentence pairs were retained; the exact ranking and selection procedure is not explicitly reported.**
  - 未报告是否采用额外去重、分数归一化、并列分数处理或其他约束。
  - 来源:Sections 4.4、5.2, pp. 9411–9413。

- [x] **模型大小和 COMET-QE checkpoint 是否明确报告?**

  | Parameter | Value |
  |---|---|
  | NMT framework | fairseq |
  | NMT architecture | Vanilla Transformer with GeLU |
  | NMT parameter count | Not reported |
  | Encoder/decoder layers | Not reported |
  | Hidden dimension | Not reported |
  | Attention heads | Not reported |
  | FFN dimension | Not reported |
  | COMET-QE backbone | XLM-RoBERTa |
  | COMET-QE checkpoint | Not reported |
  | COMET evaluation checkpoint | Not reported |
  | Tokenizer | SentencePiece |
  | Vocabulary size | 16K per language(En-De、En-Sw) |

  - Section 4.3 明确说明 NMT 使用 vanilla Transformer 架构,并以 GeLU 替代 ReLU。
  - 没有报告 NMT 的参数数量及具体架构规模,因此不能直接认定为 Transformer-base 或约 65M 参数的模型。
  - COMET-QE 引用 Rei et al. (2020),但没有提供 checkpoint ID。
  - 不能确认其使用的 COMET-QE 与我们的 `Unbabel/wmt22-cometkiwi-da` 相同。
  - 同样,作为下游评价指标的 reference-based COMET 也没有报告具体 checkpoint。
  - 文中明确列出的 BLOOM 560M、1.1B 和 1.7B 是用于 perplexity-based pruning 的模型规模,不是下游 NMT 或 COMET-QE 的参数量。
  - 来源:Sections 4.3–4.4、Table 1。

- [x] **Section 5.2 声称 Swahili 的 QE 一贯优于 Random,但 Table 4 保留 10% 时 FLORES 为 15.3 vs 15.7,应该以哪个为准?**

  **应以 Table 4 的具体数值为准。Section 5.2 的概括存在例外。**

  Table 4:En→Sw,COMET-QE 与 Random 的 BLEU 比较。

  | Test set | Retention | Random | COMET-QE | Difference |
  |---|---:|---:|---:|---:|
  | FLORES | 10% | 15.7 ± 0.8 | 15.3 ± 0.9 | −0.4 |
  | FLORES | 30% | 26.7 ± 0.9 | 27.6 ± 1.0 | +0.9 |
  | FLORES | 50% | 28.7 ± 1.0 | 29.8 ± 1.0 | +1.1 |
  | MAFAND-MT | 10% | 14.4 ± 0.6 | 16.8 ± 0.8 | +2.4 |
  | MAFAND-MT | 30% | 22.0 ± 0.8 | 24.5 ± 0.8 | +2.5 |
  | MAFAND-MT | 50% | 23.6 ± 0.8 | 25.4 ± 0.9 | +1.8 |

  Table 4:COMET-QE 相对于 Random 的各项指标变化。

  | Test set | Retention | ΔBLEU | ΔchrF++ | ΔCOMET |
  |---|---:|---:|---:|---:|
  | FLORES | 10% | −0.4 | −0.5 | +2.45 |
  | FLORES | 30% | +0.9 | +0.7 | +1.73 |
  | FLORES | 50% | +1.1 | +1.2 | +1.38 |
  | MAFAND-MT | 10% | +2.4 | +2.5 | +4.40 |
  | MAFAND-MT | 30% | +2.5 | +2.1 | +2.94 |
  | MAFAND-MT | 50% | +1.8 | +1.6 | +1.52 |

  - 在 FLORES 保留 10% 的条件下,COMET-QE 的 BLEU 和 chrF++ 均低于 Random,但下游 COMET 高于 Random。
  - 因此,Swahili 上 COMET-QE 优于 Random 是总体趋势,而非所有保留率、测试集及评价指标均成立。
  - BLEU 的 −0.4 和 chrF++ 的 −0.5 是观察到的数值差异,不能声称 COMET-QE 显著差于 Random,因为没有报告两者之间的直接显著性检验。
  - Section 5.2 将 COMET-QE、LaBSE、LASER 共同归类为 translation quality estimators,而非仅讨论 COMET-QE。
  - 正文关于 LASER 的概括同样与 Table 4 不一致:LASER 在 Swahili 的所有报告条件下均远低于 Random。
  - 正式记录:**COMET-QE generally outperforms random selection for En-Sw, but exhibits slightly lower BLEU and chrF++ on FLORES at 10% retention, despite achieving a higher COMET score.**
  - 来源:Section 5.2、Table 4, p. 9424。

- [x] **训练了多少步?是否改变过训练长度?**

  Table 1:下游 NMT 训练配置。

  | Parameter | Value |
  |---|---|
  | Maximum epochs | 30 |
  | Total optimizer updates | Not reported |
  | Warmup updates | 4,000 |
  | Effective batch size | 16K |
  | Optimizer | Adam |
  | Adam betas | (0.9, 0.98) |
  | Learning rate | 5e-4 |
  | Warmup learning rate | 1e-7 |
  | Scheduler | Inverse square root |
  | Label smoothing | 0.1 |
  | Gradient clipping | 1.0 |
  | Dropout | 0.2 |
  | Maximum sequence length | 256 |
  | Mixed precision | FP16 |
  | Hardware | 4 × A100 40GB |

  Section 4.3 报告的训练耗时:

  | Pruning rate | Retention rate | Sentence pairs(约) | Training time |
  |---|---:|---:|---:|
  | 90% | 10% | 380K | 1 hour |
  | 70% | 30% | 1.14M | 2 hours |
  | 50% | 50% | 1.90M | 3 hours |
  | 0% | 100% | 3.80M | 4.5 hours |

  - 论文仅明确报告最大训练 epoch 数为 30,没有报告各组实际完成的 epochs、optimizer updates 或累计训练 tokens。
  - 4,000 steps 是学习率 warmup 步数,不是总训练步数。
  - Table 1 表示其超参数适用于所有实验,没有报告专门针对 COMET-QE 和 Random 延长训练时间、匹配总更新步数或匹配累计 token exposure 的实验。
  - 不同保留率对应不同训练耗时,不能认为各保留率使用了相同的计算预算。
  - 即使在相同句对保留率下,不同筛选方法的平均句长也可能不同,导致每个 epoch 的 token 数及 optimizer updates 不同。
  - CAT-DIFF(1,2) 和 CAT-DIFF(1,5) 比较的是用于计算筛选分数的早期 checkpoint,并不是下游 NMT 分别只训练 2 或 5 epochs。
  - 正式记录:**Training was configured with a maximum of 30 epochs and 4,000 warmup updates. Actual optimizer updates and token exposure were not reported. No explicit training-duration-controlled comparison was presented for COMET-QE versus random selection.**
  - 来源:Sections 3、4.3、5.1,Tables 1–2。

**补充核对:Table 5 中 En→De 的 COMET-QE 退化**

Table 5:COMET-QE 与 Random 的 BLEU 比较。

| Test set | Retention | Random | COMET-QE | Difference |
|---|---:|---:|---:|---:|
| FLORES | 10% | 23.7 ± 1.0 | 16.7 ± 0.9 | −7.0 |
| FLORES | 30% | 28.8 ± 1.1 | 28.5 ± 1.1 | −0.3 |
| FLORES | 50% | 30.0 ± 1.1 | 30.7 ± 1.1 | +0.7 |
| WMT18 | 10% | 28.6 ± 0.7 | 19.4 ± 0.6 | −9.2 |
| WMT18 | 30% | 34.9 ± 0.7 | 33.9 ± 0.7 | −1.0 |
| WMT18 | 50% | 36.7 ± 0.8 | 37.8 ± 0.7 | +1.1 |

COMET-QE 相对于 Random 的各项指标变化:

| Test set | Retention | ΔBLEU | ΔchrF++ | ΔCOMET |
|---|---:|---:|---:|---:|
| FLORES | 10% | −7.0 | −7.4 | −13.07 |
| FLORES | 30% | −0.3 | −0.3 | −0.24 |
| FLORES | 50% | +0.7 | +0.3 | +2.10 |
| WMT18 | 10% | −9.2 | −8.4 | −12.65 |
| WMT18 | 30% | −1.0 | −0.7 | −0.34 |
| WMT18 | 50% | +1.1 | +0.4 | +1.76 |

- En→De 在保留 10% 时出现跨测试集、跨评价指标的一致性能退化。
- 保留 30% 时,COMET-QE 与 Random 的差距明显缩小。
- 保留 50% 时,COMET-QE 在所有报告指标上超过 Random。
- 这提供了较强的描述性证据,表明 QE-based filtering 的效果取决于 retention rate。
- 注意:这里观察到的是随数据保留率增加而缩小的性能差距,不是固定数据集下随训练步数增加而缩小的差距。
- Table 6 的 En→Fr 实验仅比较 CAT-DIFF、Random、LaBSE 和 Full,没有报告 COMET-QE。因此不能将 Chimoto et al. (2024) 引用为 En→Fr 上 COMET-QE 退化的直接证据。
- 原文 COMET 采用 0–100 分数尺度,上述 ΔCOMET 不应直接与 Waldendorf et al. (2025) 的 0–1 尺度混用。

**补充核对:Appendix A 的数据质量分析**

- 主要实验使用从各原始语料随机抽取的 3.8M 句对。
- En→De:WMT19,原始约 38M 句对。
- En→Fr:WMT15,原始约 38M 句对。
- En→Sw:WMT22,Section 4.1 报告约 22M 句对,Section 5.5 则报告约 23M,属于原文近似规模表述差异。
- En→Sw 语料由网络数据自动对齐获得,使用过 LASER。
- Appendix A 的 Figure 7 显示 German 和 French 的句长方差较大,存在超过 500 words 的异常长句,而 Swahili 不存在超过 250 words 的句子。
- Figure 8 显示 German 和 French 包含语言识别不匹配的句子,而 Swahili 在其语言识别统计中没有此类样本。
- 作者据此推测,不同候选语料的数据质量及预处理差异可能解释 QE 方法在不同语言对上的表现差异。
- 但没有通过控制实验隔离语言、噪声比例、句长分布和 lexical diversity 的影响。
- 论文没有明确报告统一的发布前去重算法或近似去重阈值。
- Section 6 和 Figure 6 分析了 Random、CAT-DIFF 和 LaBSE 的 unique words、rare words 及 mean word frequency,但没有对 COMET-QE 筛选组进行同样的词汇多样性分析。

**与 Waldendorf et al. (2025) 的对照**

| Property | Chimoto et al. 2024 | Waldendorf et al. 2025 |
|---|---|---|
| QE method | COMET-QE | COMET-KIWI |
| QE checkpoint | Not reported | Not reported |
| Language pairs tested with QE | En-De, En-Sw | En-De, En-Ar |
| Minimum retention | 10% | 1% |
| En-De degradation | BLEU、chrF++、COMET | 主要为 COMET |
| Near-deduplication | Not reported | Aggressive near-deduplication |
| Training budget | Maximum 30 epochs | 100K updates |
| QE-specific lexical diversity analysis | No | No |
| Controlled contextual diversity experiment | No | No |

**对我们论文的意义**

- Chimoto et al. (2024) 是证明 aggressive QE-based filtering 可能不如 Random 的重要相关研究,尤其是 Table 5 中 En→De 保留 10% 时出现的明显退化。
- 与 Waldendorf et al. (2025) 相比,该论文在 En→De 上观察到跨评价指标的一致退化,而不仅是 neural metric degradation。
- 两篇论文均没有明确报告 QE checkpoint,因此不能视为对同一个 CometKiwi-22 checkpoint 的严格重复验证。
- Chimoto et al. 将语言对差异归因于候选数据质量及预处理差异;Waldendorf et al. 则提出人工 QE 监督训练语种覆盖差异这一解释。
- 两种解释都没有直接检验 QE 筛选是否通过压缩 lexical coverage 和 contextual diversity 导致下游 NMT 性能下降。
- Chimoto et al. 已经对其他筛选方法进行了 lexical diversity 的描述性分析,因此不应声称此前研究完全没有关注数据筛选与词汇多样性的关系。
- 更准确的研究缺口是:已有工作没有在 QE 筛选条件下,通过匹配训练预算、词汇覆盖及词汇频率分布,直接比较重复暴露与上下文多样性的独立作用。
- 我们的"重复那一半 vs 窄词表多样组"实验可以针对这一机制进行直接检验。

**Related Work 英文表述(草稿)**

Chimoto et al. (2024) observed substantial degradation from COMET-QE-based data pruning for English–German when retaining only 10% of the candidate corpus, with reductions of 7.0 and 9.2 BLEU points relative to random selection on FLORES and WMT18, respectively. These differences diminished at higher retention rates, with COMET-QE outperforming random selection at 50% retention. The authors attributed language-dependent differences partly to corpus quality and preprocessing characteristics. Although they examined lexical diversity for selected pruning methods, they did not directly analyze lexical coverage or contextual diversity in COMET-QE-selected subsets. Moreover, the exact QE checkpoint, actual optimizer updates, and cumulative token exposure were not reported, leaving the mechanisms underlying aggressive quality-based filtering failures unresolved.

---

## Claude 的补充(2026-11-27)

1. **两篇论文在 En→De 保留 10% 这一点上结论相反。** Waldendorf:COMET-KIWI 比随机高 3.0 spBLEU;Chimoto:COMET-QE 比随机低 7.0–9.2 BLEU。打分器不同(COMET-QE 2020 对 COMET-KIWI)、池子不同(WMT19 随机抽 3.8M、未报告去重,对 ParaCrawl 52M、近似去重)、训练量不同。所以"QE 从输给随机变成赢过随机"的转折点取决于池子和打分器,不能从文献里借一个数;要回答 Peter et al. 那条反驳,W2(在我们自己的池子里沿保留比例扫一遍)有必要。
2. **这篇可以用来削弱"人工打分监督覆盖"的解释(推断,待核实)。** En→De 在 COMET 系列的人工打分训练数据里,却在 10% 时大败;En→Sw 多半不在 COMET-QE 的训练数据里(没有核实),QE 反而总体胜随机。两篇论文合起来看,"有没有人工监督"预测不了 QE 筛选的成败。这对我们回应"En-Fr 没有人工监督,所以是判分不准"有用,但引用前要核实 COMET-QE(Rei et al. 2020)的训练语言对,而且两篇用的不是同一个检查点。
3. **他们的"池子噪声"解释,我们已经测过。** §3.2 第 3 项(10 月 19 日):长度比离群重复那一半 2.14%、多样 3.13%、随机 UN 5.96%;外语行两半都不到 0.4%。重复那一半反而更干净,所以噪声解释不了我们两半之间的差距。
4. **± 和显著性的口径。** 表里每格是单次训练,± 是测试集 bootstrap 区间的半宽,加粗是 CAT-DIFF 对其他方法的检验;COMET-QE 对 Random 没有直接检验。En→De 10% 的两个区间不重叠(23.7 ± 1.0 对 16.7 ± 0.9),可以写"差距远大于 bootstrap 区间",不能写"显著"。也不能和我们 5 个训练种子的标准差直接比。
5. **训练预算。** 最多 30 个 epoch,所以保留 10% 的组更新步数大约只有全量的十分之一;同一保留率下各方法句数相同,但 token 数未必相同。我们固定 10,000 步、各组句长分布对齐,这一点更干净。学习率方面他们是从头训练(5e-4),对"激进学习率"质疑没有参考价值。
6. **Related Work 草稿。** 事实准确。建议补三处限定:从头训练;每个语言对是随机抽取的 3.8M 句对子集;COMET-QE 是 Rei et al. (2020) 的参考无关模型,不是 CometKiwi。"substantial degradation ... relative to random selection"在这里是对的(表里有随机基线)。

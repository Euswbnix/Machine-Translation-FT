# Erdmann & Gwinnup 2018 + Azpeitia et al. 2018(作者阅读笔记,2026-11-27)

以下是作者本人的阅读结果,原样保存。文末"Claude 的补充"是讨论意见,不是原文内容。

## AFRL 的 coverage 定义

- Coverage 不是简单的 vocabulary coverage,而是衡量所选语料的词汇特征频率分布与参考语料分布的匹配程度。
- 使用对数函数提供递减的边际收益,并对超过参考频数的特征施加 oversaturation penalty。
- 每个特征的饱和阈值由缩放后的参考语料频数决定,而非统一的固定阈值。
- 2018 年论文使用 BPE(2000 merges),并在特征相关描述中提到 1–4 grams,但没有明确说明 coverage 的具体 n-gram 阶数、源端和目标端的聚合方式,以及参考语料 T 的具体构成。
- 其引用的 Gwinnup et al. (2016) 使用 unigram 和 bigram,并以 2013–2015 年新闻测试集作为参考语料,用于英语单语语言模型数据选择。不能直接认定 2018 年沿用了相同配置。
- Coverage 达到饱和后,额外出现的特征不再产生覆盖收益,并增加惩罚项。通过 greedy selection 决定句子的保留或移除。
- 来源:https://aclanthology.org/W18-6475.pdf,Sections 3–5。

## Vicomtech 的 n-gram saturation 定义

- 使用源端(德语)的 unigram、bigram 和 trigram,即 n = 1, 2, 3。
- 按照 STACC.OOV 翻译质量评分从高到低遍历候选句对。
- 使用 Patricia trie 记录此前遍历过的所有 n-grams。
- 已出现过的 n-gram 即被视为已覆盖,不设置额外的频数阈值。
- 根据正文描述,saturation score 为当前句子中新出现的 n-grams 占全部 n-grams 的比例。
- 最终评分为 STACC.OOV score × n-gram novelty score。
- 饱和后采取降分而非直接删除的方式;完全没有新 n-grams 的句子按照上述定义得分为 0。
- 注意:原文 Equation 4 的分母写的是已出现的 n-grams 数量,与正文描述的全部 n-grams 数量不一致,且存在除零问题。应当将 new / total 标注为根据正文重构的定义,而非已经核实的原始代码实现。
- 来源:https://aclanthology.org/W18-6473.pdf,Section 2.3。

## 两篇论文报告的实验效果

Koehn et al. (2018), Table 4:

| System | SMT 10M | SMT 100M | NMT 10M | NMT 100M |
|---|---:|---:|---:|---:|
| AFRL cvg-large | 21.9 | 25.2 | 13.8 | 30.2 |
| AFRL cvg-small | 21.9 | 22.9 | 13.5 | 21.1 |
| AFRL cvg-mix | 22.5 | 25.2 | 19.8 | 30.1 |
| AFRL cvg-mix-meteor | 23.4 | 25.3 | 27.1 | 30.3 |
| AFRL cyn-mix | 22.4 | 25.0 | 25.1 | 29.6 |
| Vicomtech | 23.2 | 25.9 | 26.4 | 30.4 |
| Vicomtech ngsat | 23.3 | 25.8 | 25.6 | 24.9 |

- AFRL 的 cvg-large 和 cvg-small 在 NMT 10M 上分别取得 13.8 和 13.5 BLEU,确认无误。
- AFRL 作者指出,加入 METEOR 翻译质量评价能够显著改善小规模 NMT 表现,混合针对不同数据规模设计的 coverage score 也更加有效。
- AFRL 的预过滤移除了约 90% 的候选语料,可能同时移除了有价值的训练样本。
- 因此,AFRL 的结果不能直接归因于 coverage optimization 本身,因为存在翻译质量、频率分布、预过滤及语料规模等混杂因素。
- Vicomtech 的 ngsat 变体使 NMT 100M 从 30.40 下降至 24.91,下降 5.49 BLEU,确认无误。
- Vicomtech 作者推测,NMT 可能需要更多 contextual variants,而 n-gram saturation filtering 可能降低这些有价值的上下文变体的排名。
- 论文 Footnote 4 还记录了另一种解释:SMT 更依赖具体 n-gram 的覆盖,而 NMT 可以通过 embeddings 获得一定的泛化能力;saturation filtering 可能改变训练数据分布,产生 data imbalance。
- 上述解释均为机制假设,论文未通过控制实验建立因果关系。
- 官方实验结果来源:https://aclanthology.org/W18-6453.pdf,Table 4。
- Vicomtech 作者解释来源:https://aclanthology.org/W18-6473.pdf,Section 4。

## 哪个定义可以应用于"重复那一半 vs 窄词表多样组"的实验

- Vicomtech 的 n-gram novelty 思想更加适合。
- AFRL 的 coverage 同时涉及参考频率分布匹配和过饱和惩罚,不能直接作为独立的上下文多样性指标。
- Vicomtech 的方法能够区分词汇重复与已有词汇在不同上下文中的重新组合。
- 假设重复组为 A ⊎ A,多样组为 A ⊎ B,其中 ⊎ 表示保留重复样本的语料拼接。
- 定义 Novelty_n(B | A) = |G_n(B) \ G_n(A)| / |G_n(B)|,其中 G_n 表示语料中不同 n-gram types 的集合。
- 对于重复组,Novelty_n(A | A) = 0。
- 对于窄词表多样组,预期 unigram novelty 接近 0,但 bigram 和 trigram novelty 显著大于 0。
- 建议分别报告 n = 1, 2, 3 的 novelty,而非仅报告合并分数。
- 同时报告 n-gram diversity ratio,即 unique n-gram types / total n-gram occurrences。
- 上述指标是基于 Vicomtech 思想构造的、与语料内部顺序无关的统计指标,并非原始过滤算法。
- 实验应尽可能匹配 training token budget、unigram vocabulary coverage、unigram frequency distribution、句长、领域分布及翻译质量,以减少混杂因素。

## 对本文创新性的影响

- Azpeitia et al. (2018) 已经提出 contextual variants 可能对 NMT 训练有益,但没有通过匹配词汇覆盖与训练预算的控制实验来验证该假设。
- 本文的潜在贡献不是首次提出数据多样性的重要性,而是在控制词汇覆盖和训练预算的条件下,通过重复样本与窄词表多样样本的直接比较,尝试解耦 lexical coverage、repeated exposure 和 contextual diversity 对 NMT 性能的影响。
- 是否已有后续研究完成类似的解耦实验,仍需进一步核查文献。

## Related Work 英文表述(草稿,结果出来后再定)

Azpeitia et al. (2018) observed that incorporating source-side n-gram saturation into parallel corpus filtering reduced NMT performance by 5.49 BLEU points in the 100M-word setting. They hypothesized that this degradation might result from the suppression of useful contextual variants, but did not isolate this effect experimentally. Motivated by this observation, we design a controlled comparison between repeated training examples and contextually diverse examples under matched training budgets and comparable lexical coverage, aiming to disentangle the effects of lexical coverage, repeated exposure, and contextual diversity on neural machine translation performance.

---

## Claude 的补充(2026-11-27,供 12 月下旬定 E0.6 时讨论)

1. **和 E0.6 现有设计的关系。** 预注册草稿的 OD 1 推荐 (b)(源端和法语端词型都在 ±2% 内),只能配 OD 2 = c2。这一步相当于把 unigram types 对齐了;笔记建议的 bigram、trigram novelty 正好是对齐之后还剩下的差别,适合作为"跑之前在各组上实测"的描述量登记进预注册,而不是作为匹配量。建议 n 取 1、2、3、4,再加 8,和我们的"重复程度"(8 词片段)接上。
2. **"A ⊎ A"的模型要调整。** 重复那一半里完全重复的源句只占 8.45%,大部分"重复"是不同句子之间共享的 8 词片段。所以 novelty 最好在每个组内部算(例如:某个 n-gram 出现位置中,其 type 在本组其他句子里也出现过的比例;以及每个 n 的 type/token 比),而不是相对一个参考集 A 算。各组句数和句长分布已对齐,type/token 比可以直接比较。
3. **unigram 频率分布目前没有对齐。** OD 1 只对齐词型个数,两组词型数相同,频率分布(例如 unigram 熵、Zipf 斜率)仍可能不同。建议作为描述量一并报告;把它也加进匹配条件,`div_narrow` 很可能构建不出来。
4. **Related Work 草稿的两处措辞。** (a) Vicomtech 的 −5.49 出现在 100M 词;10M 词时 ngsat 只从 26.4 降到 25.6(−0.8);设置是过滤嘈杂的 ParaCrawl、从头训练 Marian 的 RNN NMT。写清楚设置和两个规模,避免被说以偏概全。(b) "disentangle lexical coverage, repeated exposure, and contextual diversity":E0.6 能分开的是覆盖(rep vs div_narrow)和完全重复的源句(rep vs rep_dedup);"共享 n-gram 的重复"和"上下文多样性"是同一股特征的两种说法,E0.6 不会把它们分成两股。建议写成"coverage and exact duplication",或者等结果出来再定。
5. **"后续研究是否已完成类似解耦"** 这一条,清单第三组(WMT19 三篇系统论文、Cettolo et al. 2014)和第二梯队里的 Xu et al. 2020(n-gram 覆盖折扣)正好覆盖一部分;读完后如果仍不确定,再做一次专门的文献检索。

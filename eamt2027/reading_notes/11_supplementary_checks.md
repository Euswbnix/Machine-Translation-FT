# 补充文献核对(作者阅读笔记,2026-11-27)

以下是作者本人的阅读结果,原样保存。对应 `06_tan_2024.md`、`07_finkelstein_2024.md`、`08_fujita_de_mello_takada_2025.md` 里留下的三个小尾巴。文末"Claude 的补充"是讨论意见。

## 1. Tan et al. 2024:中文 BLEU 的分词方式

- [x] Section 4.3, Footnote 4, p. 179 明确给出了完整的 SacreBLEU signature:

  `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.3.1`

  | Parameter | Value |
  |---|---|
  | Evaluation library | SacreBLEU 2.3.1 |
  | Tokenizer | 13a |
  | Case sensitivity | Case-sensitive |
  | Smoothing | Exponential |
  | References | 1 |
  | Effective order | False |
  | Evaluation text | Detokenized |

- Section 2, p. 177 明确说明,数据预处理阶段使用 Jieba 进行中文分词,使用 KyTea 进行日语分词。
- 但是,Jieba 属于数据预处理流程,不能直接认定为 BLEU 评估时使用的分词器。
- 原文报告的 BLEU tokenizer 是 `13a`,而不是中文任务通常使用的 `zh`。
- 这里存在一个值得注意的复现问题:`13a` 不会像 `zh` tokenizer 那样自动拆分连续中文字符。
- 原文没有单独说明中文评估文本是否保留了 Jieba 产生的空格,也没有提供各语言方向独立的 SacreBLEU signature。
- 因此,无法确认作者是否直接对未经分词的中文文本使用 `13a`,或者使用了预先分词的中文文本。
- Table 4 使用 FLORES-devtest,应参考 Section 4.3 的评价设置。
- Table 3 中 WMT24 official test set 的 BLEU 和 chrF 来自 OCELoT submission system,不一定采用相同配置。

**正式记录:** SacreBLEU v2.3.1 with the 13a tokenizer was explicitly reported. Jieba was used for Chinese data preprocessing, but its use during BLEU evaluation was not confirmed.

**对本文的影响:** Tan et al. 的中文 BLEU 绝对数值不宜直接与使用 `zh` 或 `flores200` tokenizer 的研究进行比较,但其 Table 4 内部不同 QE 阈值之间的相对性能变化仍具有参考价值。

来源:https://aclanthology.org/2024.wmt-1.11.pdf,Sections 2、4.3,Footnote 4。

## 2. Finkelstein et al. 2024:Table 3 的平均句长

- [x] 已重新检查正式出版 PDF 的 Table 3,p. 1358,确认数字无误。

| Direction | Dataset | Source length | Target length |
|---|---|---:|---:|
| En→De | MBR sentence-level | 37.5 | 39.8 |
| En→De | QE blob-level | 364.5 | 339.8 |
| De→En | MBR sentence-level | 77.3 | 88.3 |
| De→En | QE blob-level | 288.4 | 323.4 |

- 单位为 Moses tokenizer 计算的平均 token 数,不是字符数、原始空格分词词数,也不是 NMT 模型的 subword token 数。
- De→En sentence-level 的 77.3 / 88.3 确实是原文数值,不是 PDF 提取错误。
- 原文没有解释为什么 De→En sentence-level 样本的平均长度约为 En→De 的两倍。
- 两个翻译方向使用分别构建的 NewsPaLM 数据集,并非将同一批平行句对交换源端与目标端,因此不要求平均长度对称。
- Section 2.2 中用于构造合成 blob-level 数据的 512-token 长度限制使用 PaLM-2 tokenizer,与 Table 3 的 Moses tokenizer 统计口径不同。

根据 Table 2 的样本数量及 Table 3 的平均长度,估算整个数据集的 source + target tokens:

| Direction | Dataset | Examples | Approximate total Moses tokens |
|---|---|---:|---:|
| En→De | MBR sentence-level | 998,435 | 77.18M |
| En→De | QE blob-level | 925,829 | 652.06M |
| De→En | MBR sentence-level | 1,022,344 | 169.30M |
| De→En | QE blob-level | 769,028 | 470.89M |

- En→De 的 blob-level 数据总文本量约为 sentence-level 的 8.45 倍;De→En 约为 2.78 倍。
- 上述数量是根据四舍五入后的平均长度推算的数据集规模,不是实际累计训练 token exposure。

**关键注意事项:** Table 3 统计的是 NewsPaLM 合成数据,而不是从 WMT23 的 296M 候选句对中通过 BLEURT-QE 筛选得到的真实平行数据。后者的平均句长及 token 数量没有报告。因此,不能使用 Table 3 的 37.5 / 39.8 或 77.3 / 88.3 估算真实 BLEURT-QE 筛选组的训练 token budget。

**正式记录:** Table 3 reports average source and target lengths of the synthetic NewsPaLM datasets using the Moses tokenizer. The corresponding length statistics for the BLEURT-QE-filtered WMT23 subsets were not reported.

来源:https://aclanthology.org/2024.wmt-1.126.pdf,Section 2.5,Tables 2–3,pp. 1357–1358。

## 3. Fujita de Mello & Takada 2025:IEEE 正式出版版本

- [x] 正式出版信息及 DOI 已找到。
- [ ] IEEE 正式版全文与 arXiv 预印本之间的差异尚未核实。

**正式出版信息:**

- Title: Improving Translation Quality by Selecting Better Data for LLM Fine-Tuning: A Comparative Analysis
- Authors: Felipe Ribeiro Fujita de Mello, Hideyuki Takada
- Conference: 2025 IEEE International Conference on Big Data (BigData),Macau, China,December 8–11, 2025
- Pages: 5490–5499
- DOI: 10.1109/BigData66926.2025.11402145;IEEE document ID: 11402145

**正式版入口:** DOI https://doi.org/10.1109/BigData66926.2025.11402145 ;IEEE Xplore https://ieeexplore.ieee.org/document/11402145/ ;DBLP https://dblp.org/rec/conf/bigdataconf/MelloT25 ;arXiv https://arxiv.org/abs/2512.11388

作者 Hideyuki Takada 的个人学术主页及 DBLP 均确认该论文已正式发表于 IEEE Big Data 2025。IEEE Big Data 2025 官方 workshop schedule 也列出了该论文,因此不宜未经核实将其描述为 Big Data main-track paper。

**BibTeX:**

```bibtex
@inproceedings{fujita2025improving,
  author    = {Felipe Ribeiro Fujita de Mello
               and Hideyuki Takada},
  title     = {Improving Translation Quality by
               Selecting Better Data for LLM
               Fine-Tuning: A Comparative Analysis},
  booktitle = {2025 IEEE International Conference
               on Big Data (BigData)},
  year      = {2025},
  pages     = {5490--5499},
  publisher = {IEEE},
  doi       = {10.1109/BigData66926.2025.11402145}
}
```

**正式版核查状态:** 本次无法访问 IEEE Xplore 正式版全文,也未找到可公开读取的独立正式版副本。因此,以下问题是否已经修正仍然未知:Table III 中 Gemma 和 Mistral 的重复 BLEU/COMET 数值;Table IV 中的重复数值;Table V 的百分比分母与 Section V-D 公式不一致;WMT24、KFTT 等参考文献编号错位;998 个测试句对的具体来源及 JA→EN 的构建方式;COMET-Kiwi checkpoint、BLEU tokenizer、LoRA rank 等未报告的实验细节。

建议通过 University of Toronto Libraries 的 IEEE Xplore 馆藏权限尝试下载正式版:https://library.utoronto.ca/

**当前文献记录:** Published at IEEE Big Data 2025. Bibliographic information and DOI identified. All previously documented reporting anomalies were verified against arXiv:2512.11388v1; whether these issues persist in the IEEE version of record remains unverified.

正式引用可以使用 IEEE 会议版本,但涉及表格异常及复现问题的内部研究笔记必须注明核查版本为 arXiv v1。

---

## Claude 的补充(2026-11-27)

1. **Tan 的中文 BLEU 多半是在分好词的文本上算的(推断)。** 如果把没分词的中文直接交给 13a,一整段连续汉字会被当成一个"词",BLEU 会接近 0;他们报的是 33–36,所以译文和参考很可能都带着 Jieba 的空格。这样算出来的是词级 BLEU,而 tok:zh 是按字切的,两者不能互相比较。结论和你的一致:只用他们 Table 4 内部的相对差距(−2.6 BLEU),不引绝对值,也不和我们 en→zh 模型的 29.25(tok:zh,newstest2017)比。
2. **Finkelstein 的句长我之前怀疑抄错,现在确认原文如此,这一条关闭。** 两个方向的数据是分别构建的,不要求对称。你补的总 token 数很有用:En→De 的 blob 数据文本量是句级数据的 8.45 倍,所以他们"blob 数据从头训练更好"的比较里训练文本量差了近一个数量级。这进一步说明那篇里可用的比较只有"QE 筛选集对 MBR 句级数据"这一对,而那一对的 token 数也没报告。
3. **Fujita de Mello 的正式版:引用用 IEEE 会议版(DOI 已有),但不写成主会论文。** 写 "In 2025 IEEE International Conference on Big Data (BigData)" 即可,不加 main track 或 workshop 的定性,除非核实。正式版全文需要你用学校图书馆的权限下载;这一项留在清单上,动笔前核一次那六处是否改过。

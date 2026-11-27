# Fujita de Mello & Takada 2025(作者阅读笔记,2026-11-27)

以下是作者本人的阅读结果,原样保存。文末"Claude 的补充"是讨论意见,不是原文内容。

**论文:Improving Translation Quality by Selecting Better Data for LLM Fine-Tuning: A Comparative Analysis**

- 作者:Felipe Ribeiro Fujita de Mello, Hideyuki Takada(Ritsumeikan University)
- 发表:2025 IEEE International Conference on Big Data, pp. 5490–5499.
- arXiv:https://arxiv.org/abs/2512.11388 ;PDF:https://arxiv.org/pdf/2512.11388
- 核查版本:arXiv v1,2025 年 12 月 12 日。未核实 IEEE 最终出版版本是否修正了以下问题。

**重点:Sections IV–V, Tables I–V**

- [x] **Table III 中 Gemma 和 Mistral 多格完全相同,Table IV 也存在重复数值,是否抄错?**

  已直接检查原始 PDF 第 5–6 页,确认重复数值确实存在于原始表格,并非 PDF 文本提取或 OCR 错误。

  Table III:JA→EN,10K KFTT training samples,WMT24 evaluation。

  | Model | Method | BLEU | COMET |
  |---|---|---:|---:|
  | LLaMA-7B | Random | 13.73 | 0.6050 |
  | LLaMA-7B | TF-IDF | 21.93 | 0.6849 |
  | LLaMA-7B | FD-Score | 20.69 | 0.7073 |
  | LLaMA-7B | QuRate | 13.95 | 0.7873 |
  | LLaMA-7B | COMET-Kiwi | 25.91 | 0.7105 |
  | Gemma-7B | Random | 17.37 | 0.8095 |
  | Gemma-7B | TF-IDF | 10.09 | 0.7450 |
  | Gemma-7B | FD-Score | 20.05 | 0.8077 |
  | Gemma-7B | QuRate | 17.53 | 0.8117 |
  | Gemma-7B | COMET-Kiwi | 20.25 | 0.7761 |
  | Qwen-7B | Random | 17.47 | 0.8097 |
  | Qwen-7B | TF-IDF | 20.72 | 0.8059 |
  | Qwen-7B | FD-Score | 18.69 | 0.8083 |
  | Qwen-7B | QuRate | 20.48 | 0.7616 |
  | Qwen-7B | COMET-Kiwi | 24.46 | 0.8147 |
  | Mistral-7B | Random | 17.37 | 0.8095 |
  | Mistral-7B | TF-IDF | 20.05 | 0.8077 |
  | Mistral-7B | FD-Score | 17.53 | 0.8116 |
  | Mistral-7B | QuRate | 20.06 | 0.7450 |
  | Mistral-7B | COMET-Kiwi | 20.25 | 0.7761 |

  Table III 中完全重复的结果:Gemma Random = Mistral Random(17.37 / 0.8095);Gemma FD-Score = Mistral TF-IDF(20.05 / 0.8077);Gemma COMET-Kiwi = Mistral COMET-Kiwi(20.25 / 0.7761)。另外两组高度相似:Gemma QuRate 17.53 / 0.8117 对 Mistral FD-Score 17.53 / 0.8116;Gemma TF-IDF 10.09 / 0.7450 对 Mistral QuRate 20.06 / 0.7450。

  Table IV:EN→JA,10K KFTT training samples,WMT24 evaluation。

  | Model | Method | BLEU | COMET |
  |---|---|---:|---:|
  | LLaMA-7B | Random | 13.67 | 0.7812 |
  | LLaMA-7B | TF-IDF | 13.79 | 0.7846 |
  | LLaMA-7B | FD-Score | 13.81 | 0.7845 |
  | LLaMA-7B | QuRate | 14.11 | 0.7873 |
  | LLaMA-7B | COMET-Kiwi | 15.06 | 0.7970 |
  | Gemma-7B | Random | 17.67 | 0.8094 |
  | Gemma-7B | TF-IDF | 13.79 | 0.7846 |
  | Gemma-7B | FD-Score | 17.88 | 0.8132 |
  | Gemma-7B | QuRate | 18.03 | 0.8118 |
  | Gemma-7B | COMET-Kiwi | 18.22 | 0.8187 |
  | Qwen-7B | Random | 9.67 | 0.7088 |
  | Qwen-7B | TF-IDF | 13.02 | 0.7838 |
  | Qwen-7B | FD-Score | 13.74 | 0.7845 |
  | Qwen-7B | QuRate | 10.30 | 0.7447 |
  | Qwen-7B | COMET-Kiwi | 14.68 | 0.8017 |
  | Mistral-7B | Random | 10.70 | 0.7056 |
  | Mistral-7B | TF-IDF | 13.02 | 0.7838 |
  | Mistral-7B | FD-Score | 13.02 | 0.7838 |
  | Mistral-7B | QuRate | 14.04 | 0.7865 |
  | Mistral-7B | COMET-Kiwi | 14.47 | 0.7925 |

  Table IV 中完全重复的结果:LLaMA TF-IDF = Gemma TF-IDF(13.79 / 0.7846);Qwen TF-IDF = Mistral TF-IDF = Mistral FD-Score(13.02 / 0.7838)。

  **核查结论:**

  - 两张表共有五组完全重复的 BLEU/COMET 数值组合,涉及 11 行实验结果。
  - 不同模型、不同筛选方法同时产生精确到两位小数的 BLEU 和四位小数的 COMET 完全相同的结果,属于值得进一步核实的 reporting anomaly。
  - 尤其是 Table III 中 Gemma 和 Mistral 的大量对应数值重复或高度相似,存在表格复制或录入错误的可能。
  - 但是,没有原始预测文件、训练日志和评估代码,无法确认究竟是表格复制错误、评估流程错误,还是实际实验结果恰好相同。不应直接认定作者抄错数据,更不能据此指控数据造假。
  - Section IV-B 只报告了 seed = 42,没有报告多随机种子实验。Tables II–IV 未提供标准差、置信区间或正式显著性检验。
  - Tables III–IV 的加粗代表每个模型在对应评价指标上的最高分,不代表统计显著性。
  - Section V-B 将 10K 实验称为 full fine-tuning,但 Section IV-B 明确说明所有模型采用 8-bit quantization + LoRA。这里的 full fine-tuning 应理解为完整的 10K 实验,而非 full-parameter fine-tuning。

  正式记录:**Multiple exact BLEU/COMET duplicates were verified directly in Tables III and IV. These constitute unexplained reporting anomalies and may reflect transcription or evaluation errors, but their cause cannot be established without the original experimental outputs.**

  来源:Sections IV-B、V-B、V-C,Tables III–IV,PDF pp. 5–6。

- [x] **Table V 的百分比为什么加起来约等于 100%,与 Section V-D 的公式是否一致?**

  Table V 原始数据:

  | Method | Exclusive samples | Reported percentage |
  |---|---:|---:|
  | Random | 9,400 | 32.11% |
  | TF-IDF | 825 | 2.82% |
  | FD-Score | 8,892 | 30.37% |
  | QuRate | 9,280 | 31.70% |
  | COMET-Kiwi | 874 | 2.98% |
  | Total | 29,271 | 99.98% |

  Section V-D 将每种方法的独占样本定义为 U_i = S_i \ ∪_{j≠i} S_j,即只被方法 i 选择、未被其他任何方法选择的样本集合。原文给出的百分比公式为 P_i = |U_i| / |∪_k S_k| × 100,分母是所有筛选子集的并集。

  但根据 Table V 的具体数值反推,其实际使用的分母几乎可以确定为 Σ_k |U_k| = 9400 + 825 + 8892 + 9280 + 874 = 29271,即所有方法的独占样本数量之和。

  | Method | Exclusive samples | Recalculated percentage | Reported percentage |
  |---|---:|---:|---:|
  | Random | 9,400 | 32.11% | 32.11% |
  | TF-IDF | 825 | 2.82% | 2.82% |
  | FD-Score | 8,892 | 30.38% | 30.37% |
  | QuRate | 9,280 | 31.70% | 31.70% |
  | COMET-Kiwi | 874 | 2.99% | 2.98% |

  两处 0.01 percentage point 的差异可能来自截断或取整方式。因此,Table V 的实际计算方式与 Section V-D 的书面公式不一致。如果五种方法完全没有重叠,两种公式才会相同;但每种方法选择 10K 样本,总选择次数为 50K,而独占样本只有 29,271 个,说明存在大量重叠。

  **每种方法自身的独占样本比例** R_i = |U_i| / |S_i| × 100(每个子集 10,000 个样本):

  | Method | Exclusive samples | Within-subset exclusive ratio | Shared with ≥1 other method |
  |---|---:|---:|---:|
  | Random | 9,400 | 94.00% | 6.00% |
  | TF-IDF | 825 | 8.25% | 91.75% |
  | FD-Score | 8,892 | 88.92% | 11.08% |
  | QuRate | 9,280 | 92.80% | 7.20% |
  | COMET-Kiwi | 874 | 8.74% | 91.26% |

  **关键区别:**

  - 原文的 2.98% 表示 COMET-Kiwi 独占样本在所有方法独占样本总量中的占比。COMET-Kiwi 自身训练子集的独占样本比例实际上是 8.74%,不是 2.98%;其余 91.26% 的样本至少被其他一种筛选方法选中。
  - 不能由此推出 COMET-Kiwi 与 TF-IDF 的具体重叠率,因为原文没有报告完整的 pairwise overlap matrix。也不能将 2.98% 解释为 COMET-Kiwi 与 TF-IDF 之间只有 2.98% 的样本不同。
  - Abstract 中关于不足 3% 的数据差异仍然产生较大性能变化的表述,至少不能直接由 Table V 的统计结果支持。

  **对 diversity 的解释也需要严格区分:**

  - Table V 测量的是 inter-selector exclusivity,即不同选择方法之间的样本独占性;不是 intra-subset diversity,即同一个训练子集内部的多样性。
  - COMET-Kiwi 的独占样本较少,不代表其训练数据内部存在大量重复句子。Random 的独占样本比例较高,也不能直接证明其 vocabulary coverage 或 contextual diversity 更高。
  - Table V 不包含 unigram coverage、n-gram diversity、sentence redundancy 或 semantic diversity 的直接测量。

  正式记录:**Table V appears to normalize exclusive sample counts by their sum (29,271), rather than by the union of all selected subsets as specified in Section V-D. COMET-Kiwi's within-subset exclusive sample ratio is 8.74%, not 2.98%. The reported statistic measures inter-selector exclusivity rather than within-subset diversity.**

  来源:Section V-D、Table V,PDF p. 6。

- [x] **测试集究竟是哪一份?引用 [16] 对不上,998 对是不是 WMT24++ EN→JA 反过来使用?**

  原文存在明显的参考文献编号错误:正文"WMT24 dataset [16]"实际对应的文献条目是 Toneva et al. (2018)(example forgetting);"KFTT dataset [17]"实际是 Deutsch et al. (2025), WMT24++;[18] 才是 Neubig (2011), KFTT。因此,WMT24 数据集的正确引用很可能应为 [17],KFTT 应为 [18]。

  Table I 报告的测试集:WMT24,998 对,日语词表 20,101,英语词表 24,604。

  根据 WMT24 官方 General Machine Translation Shared Task:官方 EN→JA 方向存在,JA→EN 方向不存在(WMT24 取消了 X→English 方向的官方评估);官方 WMT24 的英语源端包含 998 个 paragraph-level segments;WMT24++ 继承同一组 998 个英语源端段落,扩展至 55 种语言及语言变体,日语 subset 为 `en-ja_JP`,没有单独的 `ja-en` subset。WMT24++ 的字段:source(原始英语文本)、target(人工 post-edited 日语译文)、original_target(原始日语参考译文)、segment_id、domain、is_bad_source。

  结合以下证据:Table I 的测试集大小恰好为 998;参考文献 [17] 实际是 WMT24++;官方 WMT24 没有 JA→EN 方向;作者同时报告两个方向;Section V-C 将 EN→JA 描述为反向实验。最合理的推测是:作者使用了 WMT24 或 WMT24++ 的 EN→JA 平行测试数据,并通过交换源端和目标端构造 JA→EN 评估集。

  需要保留的不确定性:998 个样本并不能单独证明使用的是 WMT24++(原始 WMT24 同样是 998 段);未报告具体 dataset repository、configuration name、下载版本或数据文件;未报告使用 `target` 还是 `original_target` 作为日语参考;未明确说明 JA→EN 是通过交换构造的;未报告是否移除了 `is_bad_source` 样本或 canary segment;未核实原始测试文件。

  **如果确实采用反向测试集,则存在额外的解释限制:** JA→EN 的源端是人工翻译得到的日语,而不是自然产生的原生日语文本;英语参考是产生该日语译文的原始英语文本;该评估可能存在 translationese 或 reverse-direction evaluation bias;不能将其与正式构建的 JA→EN benchmark 直接视为等价;两个方向的性能差异也不能完全归因于筛选方法或模型的语言能力。

  正式记录:**The test set likely corresponds to the 998 English-source segments from WMT24/WMT24++, with the Japanese-to-English direction potentially constructed by reversing the English-to-Japanese pairs. However, the exact dataset version and reversal procedure were not reported. The in-text citation [16] is incorrect and likely refers to bibliography entry [17], WMT24++.**

  来源:Fujita de Mello & Takada (2025), Sections IV-A、IV-C、V-C、Table I、References;WMT24 General MT 官方任务 https://www2.statmt.org/wmt24/translation-task.html ;Deutsch et al. (2025), WMT24++ https://aclanthology.org/2025.findings-acl.634/ ;官方 WMT24++ 数据 https://huggingface.co/datasets/google/wmt24pp

- [x] **COMET-Kiwi checkpoint、BLEU 实现、日语分词、LoRA rank 和学习率。**

  **QE 筛选及下游评估配置:**

  | Parameter | Value |
  |---|---|
  | QE scorer | COMET-Kiwi |
  | QE checkpoint ID | Not reported |
  | Selection method | Top-k by QE score |
  | Downstream COMET checkpoint | Not reported |
  | BLEU implementation | Not reported |
  | SacreBLEU version/signature | Not reported |
  | BLEU tokenization | Not reported |
  | BLEU case sensitivity | Not reported |
  | Japanese preprocessing | MeCab |
  | Japanese BLEU tokenizer | Not explicitly reported |
  | Text normalization | Unicode NFC |
  | Model tokenization | Hugging Face AutoTokenizer |
  | Training text lowercasing | Yes |
  | Maximum sequence length | 512 tokens |
  | Deduplication | Yes, algorithm unspecified |
  | Near-duplicate removal | Not explicitly reported |

  - Section III 明确说明对每个候选句对计算 COMET-Kiwi 分数,然后选择 top-k 样本。没有提供具体 checkpoint ID,因此不能确认是否为我们的 `Unbabel/wmt22-cometkiwi-da`。
  - 参考文献包含 Rei et al. (2022) 的 COMET-22 和 CometKiwi 论文,因此 CometKiwi-22 是可能的候选版本,但原文的引用编号存在错位,不能作为确认依据。
  - Section IV-D 没有明确报告下游 COMET 的具体版本,也没有充分交代是否为 reference-based COMET。
  - 全文没有明确提到 SacreBLEU,也没有报告 BLEU signature、n-gram order、smoothing method 或具体评估库。
  - MeCab 明确用于日语数据预处理和形态切分,但没有说明 BLEU 评估时是否继续使用 MeCab。不能将其 BLEU 自动认定为 SacreBLEU `ja-mecab`。这一缺失尤其影响 EN→JA 的 BLEU 可比性。

  **模型及训练配置:**

  | Parameter | Value |
  |---|---|
  | Models | LLaMA-7B, Gemma-7B, Qwen2-7B, Mistral-7B |
  | Model type | Pretrained instruction-tuned LLMs |
  | Exact model checkpoint IDs | Not reported |
  | Quantization | 8-bit(bitsandbytes) |
  | Fine-tuning method | LoRA |
  | LoRA rank / alpha / dropout / target modules | Not reported |
  | Trainable parameter count | Not reported |
  | Learning rate | 2e-4 |
  | Epochs | 1 |
  | Batch size | 2 |
  | Gradient accumulation | 8 |
  | Nominal effective batch size per device | 16 |
  | Optimizer | AdamW |
  | Linear warmup | 5% |
  | Maximum sequence length | 512 |
  | Precision | FP16 |
  | Random seed | 42 |
  | Hardware | NVIDIA A400 GPUs, as reported |
  | Number of GPUs | Not reported |
  | Actual optimizer updates | Not reported |
  | Cumulative training tokens | Not reported |
  | Decoding parameters | Not reported |

  - LoRA rank 没有报告,不能自行假设为 8、16、32 或 64。
  - 原文硬件名称写的是 NVIDIA A400,不应未经核实改成 A4000 或 A100。
  - Section IV-B 将 Qwen 模型称为 Qwen2-7B,但未提供准确 checkpoint ID,无法确认具体版本。
  - 所有 1K 和 10K 实验均采用 1 epoch,学习率为 2e-4。论文没有报告针对不同模型或不同筛选方法单独调节学习率。
  - 在单设备、不使用 sequence packing 的假设下,10K 样本约对应 625 次 optimizer updates;1K 样本约对应 63 次。这些只是推算值,实际步数没有报告。
  - Section IV-A 声称各方法具有相同 token sizes,但没有提供每组 token 数量或实现严格 token matching 的方法。相同句对数量、相同最大序列长度及相同 epoch 数不能自动保证累计训练 token exposure 相同。该研究严格报告的是 sentence-pair-count-matched comparison,而非已经验证的 token-budget-matched comparison。

  正式记录:**COMET-Kiwi and downstream COMET checkpoint IDs, BLEU implementation, evaluation tokenization, and LoRA rank were not reported. MeCab was used for Japanese preprocessing. All models were fine-tuned with 8-bit quantization and LoRA for one epoch at a learning rate of 2e-4, using seed 42.**

  来源:Sections III、IV-A–D,PDF pp. 3–5。

---

**补充核对:Table I 的候选池规模与保留率**

| Split | Sentence pairs | JA vocabulary | EN vocabulary |
|---|---:|---:|---:|
| Full training set | 440,288 | 146,726 | 190,063 |
| Selected subset (10K) | 10,000 | 34,215 | 41,687 |
| Selected subset (1K) | 1,000 | 9,782 | 11,305 |
| Validation | 2,000 | 18,943 | 22,771 |
| Test | 998 | 20,101 | 24,604 |

相对于 440,288 句对的名义候选池:1K 为 0.2271%,10K 为 2.2712%。

- 训练语料来自 KFTT,即京都相关 Wikipedia 文章的日英平行翻译数据。与 WMT18 ParaCrawl 高噪声候选池不同,KFTT 是专业翻译构成的平行语料。
- 论文报告删除重复句子、错位句对、空句、过度重复内容和非语言符号。没有单独报告预处理前后的候选池规模变化;上述比例属于名义保留率。
- Table I 只为 10K 和 1K 各报告一组 vocabulary size,没有区分 Random、TF-IDF、FD-Score、QuRate 和 COMET-Kiwi;无法确定该词汇统计属于哪个选择方法。因此,不能根据 Table I 判断 COMET-Kiwi 是否比 Random 保留更少词汇。

**补充核对:COMET-Kiwi 是否一贯优于 Random?**

| Direction | Model | ΔBLEU | ΔCOMET |
|---|---|---:|---:|
| JA→EN | LLaMA-7B | +12.18 | +0.1055 |
| JA→EN | Gemma-7B | +2.88 | −0.0334 |
| JA→EN | Qwen-7B | +6.99 | +0.0050 |
| JA→EN | Mistral-7B | +2.88 | −0.0334 |
| EN→JA | LLaMA-7B | +1.39 | +0.0158 |
| EN→JA | Gemma-7B | +0.55 | +0.0093 |
| EN→JA | Qwen-7B | +5.01 | +0.0929 |
| EN→JA | Mistral-7B | +3.77 | +0.0869 |

- 按原文报告的数值,COMET-Kiwi 在全部八个 model-direction combinations 中均提高 BLEU。但在 JA→EN 的 Gemma 和 Mistral 实验中,COMET-Kiwi 的下游 COMET 均低于 Random。
- 两个退化结果恰好涉及前述完全重复的实验数值,因此不能将其视为两个已经可靠验证的独立观察。
- 在 EN→JA 中,COMET-Kiwi 在四种模型上均取得高于 Random 的 BLEU 和 COMET。
- 原文关于 semantic selection 一贯更优的概括需要限定模型、语言方向和评价指标。由于缺少重复实验及显著性检验,不能使用 statistically significant improvement 描述这些差值。Section V-C 使用 statistically consistent 的表述,但未提供相应统计证据。

**Table II:1K 样本消融**

| Model | TF-IDF BLEU | Kiwi BLEU | ΔBLEU | TF-IDF COMET | Kiwi COMET | ΔCOMET |
|---|---:|---:|---:|---:|---:|---:|
| LLaMA-7B | 16.83 | 14.39 | −2.44 | 0.7149 | 0.7271 | +0.0122 |
| Gemma-7B | 16.84 | 16.63 | −0.21 | 0.7313 | 0.7371 | +0.0058 |
| Qwen-7B | 5.27 | 5.31 | +0.04 | 0.6211 | 0.6219 | +0.0008 |
| Mistral-7B | 13.34 | 12.48 | −0.86 | 0.7170 | 0.7157 | −0.0013 |

- Section V-A 声称 COMET-Kiwi 在四种模型上的 COMET 均提高或持平,但 Mistral 的 COMET 实际从 0.7170 下降至 0.7157;四种模型中有三种在 BLEU 上下降。
- 1K 实验没有 Random baseline,不能根据该表判断更低保留率下 COMET-Kiwi 与 Random 的关系。Table II 也没有明确标注翻译方向。

**与前面 QE filtering 研究的比较**

| Property | Chimoto 2024 | Tan 2024 | Finkelstein 2024 | Waldendorf 2025 | Fujita de Mello & Takada 2025 |
|---|---|---|---|---|---|
| Training regime | From scratch | LLM fine-tuning | From scratch | From scratch | LoRA fine-tuning |
| Model size | Not reported | 7B | 602M | ~63M | 7B |
| QE scorer | COMET-QE | COMETKiwi | BLEURT-QE | COMET-KIWI | COMET-Kiwi |
| Exact filtering checkpoint | Not reported | Not confirmed | Not reported | Not reported | Not reported |
| Main language pairs | En-De, En-Sw | En-Zh | En-De, De-En | En-De, En-Ar | Ja-En, En-Ja |
| Main retention rate | 10% | ~0.135% | ~0.34% | 1% | ~2.27% |
| Same-pool Random baseline | Yes | Ambiguous | No | Yes | Yes |
| Training duration | Max 30 epochs | 1 epoch | Not reported | 100K updates | 1 epoch |
| Multi-seed evaluation | Limited | No | Not reported | 3 seeds | No |
| QE-specific lexical diversity | No | No | No | No | No |
| Inter-selector overlap analysis | No | No | No | No | Partial |
| Controlled contextual diversity experiment | No | No | No | No | No |

**对我们论文的意义**

- Fujita de Mello & Takada (2025) 将 QE-based data selection 的研究扩展到了专业翻译语料上的 7B LLM LoRA fine-tuning。其候选池质量、模型预训练状态和训练机制均与从头训练的 NMT 实验存在明显区别。
- 论文在约 2.27% 保留率下报告了 COMET-Kiwi 相对于 Random 的普遍 BLEU 改善,但部分 JA→EN 条件的 COMET 出现下降。这说明 aggressive QE filtering 的负面影响不应被描述为在所有模型、训练机制和语言方向上必然发生。
- 不过,Tables III–IV 存在多组完全重复的实验结果,且没有提供多种子验证,因此不适合将其作为高可信度的核心定量证据。Table V 存在明确的公式与实际统计口径不一致问题。
- 该论文分析的是不同筛选方法之间的独占样本,而不是训练子集内部的 lexical diversity 或 contextual diversity。不能将 COMET-Kiwi 的 2.98% 描述为其词汇覆盖率、重复率或内部样本多样性。
- 作者没有比较各筛选子集的实际词汇覆盖、词汇频率分布、n-gram novelty 或累计训练 token exposure。测试集方向及 BLEU 实现未充分报告。
- 对我们而言,更值得借鉴的是多模型、多筛选器的比较设计,而不是直接引用其关于不足 3% 的数据差异产生显著性能变化的结论。

**文献使用建议**

- 可作为 QE-based selection 在 LLM fine-tuning 场景中的相关工作。
- 不建议作为 aggressive QE filtering 失效的核心证据。
- 不建议将 Gemma 和 Mistral 的 COMET 下降当作两个独立验证的实验结果。
- 不建议引用 Table V 的 2.98% 作为数据多样性指标。
- 如果后续获得修正后的 IEEE 版本、原始预测文件或作者公开的评估代码,应重新核查上述异常。

**Related Work 英文表述(草稿)**

Fujita de Mello and Takada (2025) compared COMET-Kiwi with lexical, geometric, and random data selection strategies for LoRA-based fine-tuning of four 7B language models on Japanese–English parallel data. Their experiments generally favored COMET-Kiwi in BLEU, although improvements were not consistent across downstream COMET scores. The study also examined exclusive samples across selection methods, but this statistic characterizes inter-selector overlap rather than within-subset lexical or contextual diversity. Consequently, the relationship between quality-based selection, vocabulary coverage, contextual diversity, and repeated exposure remains unresolved. Our experiments address these factors through controlled comparisons under matched training budgets.

**内部文献可靠性备注(不建议直接放入 Related Work)**

The publicly available preprint contains multiple unexplained duplicate BLEU/COMET results across distinct experimental conditions, an inconsistency between the uniqueness formula and the percentages reported in Table V, and insufficient documentation of the evaluation dataset and metric implementations. Its quantitative findings should therefore be interpreted cautiously until the original experimental outputs or a corrected version become available.

---

## Claude 的补充(2026-11-27)

1. **这是文献里最接近"反例"的一篇,处理方式我同意你的建议。** 它报告 COMET-Kiwi 最顶端切片在 LLM 微调里 BLEU 全面胜过随机,和我们的主张二方向相反。Related Work 里中性地引、写明它的设置(单一领域的专业翻译池、打分前已去重并去掉"过度重复"的内容、LoRA 一个 epoch、单种子),不在正文里评论它的表格异常。真正的回应是 E1.0:同样是 LLM 微调,但有多个种子、同尺寸随机对照,而且重复和多样分开比。它的池子打分前就去掉了重复内容,所以它和主张一不冲突:那个设计把我们研究的现象事先压掉了。
2. **Table V 其实能推出一个下界,虽然推不出确切重叠率。** COMET-Kiwi 的 10,000 句里有 9,126 句至少被另一种方法选中;而 Random、FD-Score、QuRate 三种方法各自"与别人共享"的句子合计最多 600 + 1,108 + 720 = 2,428 句。所以 COMET-Kiwi 至少有 9,126 − 2,428 = 6,698 句也在 TF-IDF 的前 10,000 里。也就是说,在这个池子里 COMET-Kiwi 和 TF-IDF 的选择至少重合三分之二(前提是 Table V 的独占计数本身没错;这篇的表格有别的异常,所以只能当作提示)。有意思的地方是:一个纯词汇的选择器和 QE 模型选出了大体相同的句子。这提示 QE 最顶端的选择可能很大程度上由表面的词汇或长度特征决定,和我们"顶部段词表窄、重复多"的观察是同一类现象。不建议写进论文,除非拿到可靠版本。
3. **对 E1.0 设计的三点用处。**
   - 学习率:他们 LoRA 用 2e-4,Tan 用 2e-5(微调方式不明),两篇都只用一个学习率、没调过。E1.0 第一阶段挑学习率的网格里应当包含 2e-4,这样结果可以和文献对上;而我们在多个学习率下比较,本身就是文献里没有的。
   - 测试集方向:E1.0 的主结果用 WMT24++ 的 en-fr_FR,方向是英语原文到法语,是正向的,没有他们"把测试集反过来用"的问题。
   - 你查到的 WMT24++ 字段要写进 E1.0 的预注册:参考译文用 `target`(人工后编辑版)还是 `original_target`;`is_bad_source` 的段落要不要去掉。这两项现在的草稿里没有,不定下来就等于留了事后选择的余地。
4. **评测实现要写全。** 他们没写 BLEU 实现和日语分词方式,这正是别人没法和他们比的原因。我们每个 BLEU 都带 sacreBLEU 签名(en→zh 那次已经这样做了),E1.0 的预注册里把指标的检查点和签名都固定下来。
5. **投稿前再查一次 IEEE 正式版。** 你核的是 arXiv v1。如果正式版改了表,引用的数字要跟着改;这件事放进"动笔之前"的清单。
6. **Related Work 草稿。** 事实准确,语气合适。建议补半句设置说明:"on a single-domain pool that was deduplicated and filtered for excessive repetition before scoring, with a single seed"。"geometric"指 FD-Score 的话建议直接写方法名,免得读者猜。

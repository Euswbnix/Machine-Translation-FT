# 待读论文与核对清单(2026-11-27)

按"什么时候之前必须读完"分组。已读完并给过核对结果的两篇(Saunders & DeNeefe 2024、Peter et al. 2023)不再列。
每篇的背景、关键数字和"审稿人会怎么用它",见 `reading_list.md` 第 2 节同名小节,这里只列读哪里、核对什么。
读完一篇,把每个核对项的答案告诉我(一句话即可,写"原文没写"也算答案),我写进书单和计划。

**每篇都要回答的三个问题**(页边各写一行):
1. 我们的观察是新的吗?它报过"QE 顶部不如随机"或"顶部冗余"吗?在什么设置下(从头训练还是微调、保留多少、什么打分器)?
2. 他们认为原因是什么?是测出来的,还是一句猜测?
3. 审稿人会引哪张表、哪一节来反驳我们?

**每篇都顺手记一项(新增)**:如果他们做了微调,记下学习率、步数或 epoch 数,以及有没有换过学习率。我们最大的风险是"效应只在激进学习率下才有",这些数字决定学习率对照怎么设计、论文里怎么回应。

---

## 一、定 E0.6 之前(12 月下旬开工时第一批)

### 1. Erdmann & Gwinnup 2018 + Azpeitia et al. 2018(只读定义那一节,共约 1 小时)
- AFRL "Coverage and Cynicism":<https://aclanthology.org/W18-6475/>
- Vicomtech "STACC, OOV Density and N-gram Saturation":<https://aclanthology.org/W18-6473/>
- **为什么先读**:E0.6 要定"覆盖用什么量、按什么规则选"(预注册第 1、2 项),结果出来后不能再换。最好借用现成定义,不要自己发明。
- **核对**:
  - [x] AFRL 的"coverage"怎么算:按词还是 n-gram、源端还是目标端、相对什么词表、有没有阈值。
  - [x] Vicomtech 的"n-gram saturation"怎么算:n 取几、按什么顺序累计、饱和后怎么处理(丢弃还是降分)。
  - [x] 两篇各自报告的效果:Koehn 2018 的 Table 4 里 AFRL 的 cvg 系统 NMT 10M 只有 13.8 / 13.5,Vicomtech 的 ngsat 变体 NMT 100M 低 5.5。确认原文对这些结果怎么解释。
  - [x] 两个定义里,哪一个能直接套到我们"重复那一半 vs 窄词表多样组"的对照上。
- **已读完(2026-11-27)**:作者笔记和讨论意见见 `reading_notes/01_erdmann_gwinnup_2018_azpeitia_2018.md`。

### 2. Waldendorf et al. 2025(Findings of EMNLP,约 2 小时)
- <https://aclanthology.org/2025.findings-emnlp.495/>
- **重点**:Sec 4.4–5,Tables 1、4、13、14,App C、F、G.2
- **核对**:
  - [x] 用的是哪个 CometKiwi 检查点(原文没写的话记"没写"),和我们的 CometKiwi-22 是否同一个。
  - [x] App C 的近似去重是否也用于 En→De,还是只用于 En→Ar。
  - [x] Sec 5.1 说保留 1% 时 CometKiwi"两个语言对都更差";核对 Table 13:En→De 相对随机是不是只有 COMET 更差,chrF++ / spBLEU 持平。
  - [x] CometKiwi-22 的人工打分训练数据是否覆盖 En-Fr、En-De(他们用这一点解释语言对差异)。
  - [x] App F:随机组每个种子重抽,具体怎么抽。
  - [x] App G.2:差距随训练步数缩小的幅度。
- **已读完(2026-11-27)**:作者笔记和讨论意见见 `reading_notes/02_waldendorf_2025.md`。要点:CometKiwi-22 的人工打分训练数据不覆盖 En-Fr。

### 3. Koehn et al. 2018(WMT18 过滤任务报告,约 1 小时)
- <https://aclanthology.org/W18-6453/>
- **重点**:S3–S6,Tables 1、4、6
- **核对**:
  - [x] 池子有多少句对(原文没写就记"没写",不要凭记忆引)。
  - [x] 翻译方向是否明确写了 de→en;NMT 是什么模型(只说 Marian v1.5 默认设置的话,不要叫它 Transformer)。
  - [x] Sec 6.3 说六个系统,图例列了七个:哪个多出来。
  - [x] 发布前去重是"完全重复"还是也去了近似重复。
  - [x] "cvg"系统指的是不是 AFRL 的词表覆盖方法(和第 1 篇对照)。
- **已读完(2026-11-27)**:作者笔记和讨论意见见 `reading_notes/03_koehn_2018.md`。

### 4. Chimoto et al. 2024(Findings of ACL,约 1.5 小时)
- <https://aclanthology.org/2024.findings-acl.560/>
- **重点**:Sec 4–5,Tables 4–6,App A(他们自己的方法可略读)
- **核对**:
  - [x] Tables 4–6 的加粗代表什么(显著性?最好?),看 PDF;"±"是 bootstrap 区间还是种子波动。
  - [x] COMET-QE 保留的是高分一端吗(原文有没有明说)。
  - [x] 模型大小、COMET-QE 检查点有没有写。
  - [x] Sec 5.2 说 Swahili 里 QE 一贯胜随机,但 Table 4 保留 10% 的 FLORES 是 15.3 vs 15.7:以哪个为准。
  - [x] 训练了多少步、有没有换过训练长度。
- **已读完(2026-11-27)**:作者笔记和讨论意见见 `reading_notes/04_chimoto_2024.md`。

### 5. Lo et al. 2018 + Bernier-Colborne & Lo 2019(NRC 两篇,共约 1 小时 45 分)
- <https://aclanthology.org/W18-6481/>、<https://aclanthology.org/W19-5434/>
- **重点**:2018 读 Sec 2.1、2.4、3.2,Tables 1–3;2019 读 Sec 2.3、3.1,Tables 4–5
- **为什么**:这是"排名最前的句子冗余"的最早说法,也是任何补救方法必须对比的基线(bicov、覆盖重排)。
- **核对(2018)**:
  - [x] 10M / 100M 词对应多少句对、占池子多少、是否按英语词计。
  - [x] bicov 的"平均 +2.44 / +1.61"是什么的平均。
  - [x] bicov 丢掉句子后怎么补足预算;评测方向。
  - [x] bicov 只在 YiSi-1 和 HMM 上测过,还是也在别的打分器上测过。
- **核对(2019)**:
  - [x] 池子大小、Table 5 用的组织方系统(原文没写的话,去 Koehn et al. 2019 任务报告 <https://aclanthology.org/W19-5404/> 查)。
  - [x] 1M / 5M 词是否按英语端计。
  - [x] Table 4 的"Zipporah*"一行是什么设置。
  - [x] 覆盖重排(不带新二元组的句对打八折)有没有任何消融。
- **已读完(2026-11-27)**:作者笔记和讨论意见见 `reading_notes/05_lo_2018_bernier_colborne_2019.md`。

**第一组(定 E0.6 之前)到此全部读完。**

---

## 二、定 E1.0(LLM 臂)之前

### 6. Tan et al. 2024(UvA,WMT24,约 45 分钟)
- <https://aclanthology.org/2024.wmt-1.11/>
- **重点**:Sec 3、5.2,Table 4
- **核对**:
  - [x] "100k (Cleaned)"到底是什么:Table 4 图注说随机抽样,Sec 5.2 正文说是 QE 分低于 80 的样本。
  - [x] "53k > 87"是否包含"> 89"的句对。
  - [x] 过滤用哪个 QE 模型版本;全参数还是 LoRA。
  - [x] 学习率 2e-5、1 个 epoch 是否就是全部设置。
- **已读完(2026-11-27)**:作者笔记和讨论意见见 `reading_notes/06_tan_2024.md`。还差一小项:Section 4.3 里中文 BLEU 的 sacreBLEU 签名(是不是 tok:13a)。

### 7. Finkelstein et al. 2024(WMT24,约 1 小时)
- <https://aclanthology.org/2024.wmt-1.126/>
- **重点**:Sec 4.1,Tables 4、6、7、16,Fig 2(合成数据部分可略读)
- **核对**:
  - [x] QE 筛选集的确切大小(1M / 296M 约 0.34% 是笔记算的)。
  - [x] 池子来源:是否只点名了 Europarl、News Commentary、Rapid,有没有 UN 这类模板化语料。
  - [x] BLEURT-QE 检查点、是否纯 top-k、种子、步数、学习率。
  - [x] Table 16"微调对数据量不敏感"用的学习率和步数。
- **已读完(2026-11-27)**:作者笔记和讨论意见见 `reading_notes/07_finkelstein_2024.md`。待再核一个数:Table 3 的 De→En 句级平均长度(77.3 / 88.3)。

### 8. Fujita de Mello & Takada 2025(arXiv,约 45 分钟)
- <https://arxiv.org/abs/2512.11388>
- **重点**:Sec IV–V,Tables I–V
- **核对**:
  - [x] Table III 里 Gemma 和 Mistral 多格完全相同(随机都是 17.37 / 0.8095),Table IV 也有重复格:是不是抄错了。
  - [x] Table V 的百分比加起来约 100%,和 Sec V-D 的公式对不上:原文怎么算的。
  - [x] 测试集是哪一份(引用 [16] 对不上);"WMT24 ja-en"998 对是不是 WMT24++ en-ja 反过来用。
  - [x] COMET-Kiwi 检查点、BLEU 实现和日语分词、LoRA 秩、学习率。
- **已读完(2026-11-27,arXiv v1)**:作者笔记和讨论意见见 `reading_notes/08_fujita_de_mello_takada_2025.md`。动笔前再查一次 IEEE 正式版有没有改表。

### 9. Pang et al.,"Salute the Classic: Revisiting Challenges of Machine Translation in the Age of Large Language Models"(TACL 2025,arXiv 2401.08350)
- <https://aclanthology.org/2025.tacl-1.4/>(已正式发表在 TACL 2025,引用正式版;arXiv 版作者顺序不同)
- **为什么**:Saunders & DeNeefe 在局限性里引它,说 LLM 也有类似的术语遗忘;定 LoRA 设计前要读。
- **核对**:
  - [x] 遗忘是怎么测的、在什么模型和微调方式下(LoRA 还是全参数)。
  - [x] 是否给出"微调数据的什么属性导致遗忘";和"词表覆盖"有没有关系。
  - [x] 学习率、步数,以及有没有换过学习率。
- **已读完(2026-11-27,TACL 正式版)**:作者笔记和讨论意见见 `reading_notes/09_pang_2025.md`。

### 10. Shang et al.,"Scaling Model and Data for Multilingual Machine Translation with Open Large Language Models"(arXiv 2602.11961,至少读 Table 5 和评测设置)
- <https://arxiv.org/abs/2602.11961>(只有预印本,没找到正式发表记录)
- **为什么**:E1.0 的试跑门槛(Qwen3-1.7B-Base 的 8-shot FLORES+ en→fr 39.7 spBLEU / 83.8 COMET)出自它的 Table 5。
- **核对**:
  - [ ] Table 5 评的是 base 模型还是 instruct 模型。
  - [ ] 用的是哪一份 FLORES+(dev 还是 devtest,哪个版本)。
  - [ ] spBLEU 用哪个分词器;COMET 用哪个模型。
  - [ ] 8-shot 的示例怎么选的。

---

## 三、动笔之前(写任何关于冗余控制、覆盖控制的句子之前)

### 11. WMT19 过滤任务的三篇系统论文
- Erdmann & Gwinnup 2019,"Quality and Coverage":<https://aclanthology.org/W19-5436/>
- González-Rubio 2019,Webinterpret:<https://aclanthology.org/W19-5437/>
- Axelrod et al. 2019,"Dual Monolingual Cross-Entropy Delta Filtering of Noisy Parallel Data":<https://aclanthology.org/W19-5433/>
- **核对**:
  - [ ] 各自有没有冗余或覆盖控制,怎么做的,有没有消融。
  - [ ] 有没有说"QE 或相似度选数据需要多样性控制",或用某个统计量预测伤害。这是我们第三个主张最可能被抢先的地方。
  - [ ] Axelrod 那篇:计划里写的"DiDi、cynical selection"是否准确。

### 12. Cettolo, Bertoldi & Federico 2014,"The repetition rate of text as a predictor of the effectiveness of machine translation adaptation"(AMTA 2014)
- <https://aclanthology.org/2014.amta-researchers.13/>
- **核对**:
  - [ ] 它的"重复率"怎么定义,和我们的 8 词片段重复程度是否同一类量。
  - [ ] 它说重复率能"预测"什么:是适配的收益还是伤害。这决定我们能不能用"预测"这个词。

### 13. Karpinska et al. 2022,"DEMETR: Diagnosing Evaluation Metrics for Translation"(EMNLP 2022)
- <https://aclanthology.org/2022.emnlp-main.649/>(摘要提到 COMET 对词重复的扰动比对时体变化更敏感)
- **核对**:
  - [ ] BLEU、chrF、COMET 类指标对译文里的重复各自有多敏感。我们的次要指标(chrF、MetricX-24)选得对不对,取决于这一条。

---

## 四、第二梯队(没有硬期限,动笔前读完)

- Kirchhoff & Bilmes 2014,"Submodularity for Data Selection in Machine Translation"(EMNLP 2014):<https://aclanthology.org/D14-1014/>
- Rossenbach et al. 2018(RWTH,WMT18 过滤;摘要提到一个去冗余启发式):<https://aclanthology.org/W18-6487/>
- Lu et al. 2018(Alibaba,WMT18 过滤;摘要把语料多样性列为考察项之一):<https://aclanthology.org/W18-6482/>
- Xu et al. 2020(Volctrans,WMT20 过滤;重排时对不带新源端 n-gram 的句子降分):<https://aclanthology.org/2020.wmt-1.112/>
- Wang et al. 2018(NICT,WMT18 过滤):<https://aclanthology.org/W18-6489/>
- Pombal et al.,"Adding Chocolate to Mint: Mitigating Metric Interference in Machine Translation"(TACL 2025,arXiv 2503.08327):<https://aclanthology.org/2025.tacl-1.60/>。用某个指标选数据会让相关指标偏向这些数据,是 E1.0 不用 COMET-22 作主指标的依据
- ALMA:Xu et al.,"A Paradigm Shift in Machine Translation: Boosting Translation Performance of Large Language Models"(ICLR 2024):<https://arxiv.org/abs/2309.11674>
- Littell et al. 2018(NRC 无监督提交;可能也有 bicov 结果):<https://aclanthology.org/W18-6480/>

这一组每篇只需核对两件事:
- [ ] 有没有冗余、覆盖或多样性控制,效果如何。
- [ ] 有没有和同尺寸随机数据比。

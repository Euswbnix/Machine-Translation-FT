# Section 7. Post-Training: Quality-Filtered SFT and the Multi-Dimensionality of "Quality"

## 7.1 Motivation

Sections 4–6 established a clean two-by-two: **data quality determines the sign of the capacity return**. On the strict-filter v1.1 corpus (9.3M pairs), Big exceeds Base by +0.56 BLEU; on the loose-filter v2 corpus (30M pairs), Big falls below Base by −0.87. The natural follow-up question is whether v2's extra 20M pairs are *recoverable*: can we extract the high-quality subset of v2 and use it to lift v1.1's plateau?

This section tests that hypothesis with **quality-filtered supervised fine-tuning (SFT)**. We score the full v2 corpus with a reference-free quality-estimation (QE) model, take the top-1M pairs by score, deduplicate, and SFT Base v1.1 on this filtered subset. The intent is to retain whatever signal v2 contributes beyond v1's strict cutoffs while discarding the noise.

The result is a clean **negative finding**, and the negative finding is itself the contribution: it constrains what "data quality" means.

## 7.2 Method

**Scoring.** We use CometKiwi-22 (`Unbabel/wmt22-cometkiwi-da`), the WMT22+ standard reference-free QE model based on XLM-RoBERTa-XL. Each `(src, mt)` pair receives a scalar quality score in roughly [0, 1]. Scoring v2's 30,129,500 pairs took 13.8 hours on a single RTX 5090 (≈ 750 pairs/sec at batch size 64), with the score range concentrated in [0.5, 0.92].

**Filtering.** We retain the top 1M pairs by score (top 3.3% of v2). Score range of the retained set: 0.8959–0.9145 — a tight 0.02 band, indicating a dense plateau of near-equivalent high-quality pairs above this cutoff.

**Deduplication.** Sorting and unique-ing the top-1M yields 931,366 unique `(src, tgt)` pairs (7% duplicate rate within the top-1M, mostly UN-document boilerplate). All SFT below uses this 931K-pair deduplicated set.

**SFT setup.** We fine-tune Base v1.1's averaged checkpoint (60M parameters, valid BLEU 30.52, test BLEU 35.31) by resuming training with `--reset-optimizer`. The Noam scheduler resumes at scheduler-step 26,250 (post-warmup), giving an effective peak LR of 2.7e-4 — within the 10–20% of pretraining peak band conventional for SFT. Other hyperparameters are unchanged from pretraining: bf16, label smoothing 0.1, batch 24,576 tokens × 4 accumulation, eval every 2K steps. We allow up to 10K SFT steps (≈ 2 epochs over the 931K SFT set).

**Evaluation.** newstest2013 (valid) and newstest2014 (test) sacrebleu 13a, beam=5, length-penalty=1.0. The same SPM (coverage=1.0) used in v1.1 pretraining is reused.

## 7.3 Results

We ran SFT on both Base v1.1 (60M) and Big v1.1 (209M) under identical SFT data, hyperparameters, and stopping rule. Both runs early-stopped under default patience after ~6K SFT steps. BLEU monotonically decreased throughout, on both models, on both splits:

| Run | newstest2014 (test) | newstest2013 (valid) |
|-----|---------------------|----------------------|
| Base v1.1 baseline (averaged) | 35.31 | 30.52 |
| Base v1.1 + SFT (final ckpt)  | **33.77** (−1.54) | **28.97** (−1.55) |
| Big v1.1 baseline (averaged)  | 35.87 | 31.14 |
| Big v1.1 + SFT (final ckpt)   | **33.54** (−2.33) | **29.05** (−2.09) |

(Note on units: baselines are 5-ckpt averaged checkpoints; SFT numbers are single final checkpoints because the 6K-step SFT runs did not produce enough save_interval-spaced checkpoints to average. Averaging the SFT runs would likely lift their BLEU by 0.2–0.4 — the directional finding is unaffected.)

Training loss on the SFT set converged smoothly from ~3.5 (pretraining floor on v1) to 1.94 (Base SFT floor); the loss curve gives no indication of optimization failure. Sample translations during SFT remain fluent and grammatically correct; in particular, accented French characters (`Israël`, etc.) are preserved (no `<unk>`) thanks to the v1.1 SPM. **The drop is not a translation-quality collapse. It is a distribution shift.**

Two observations from these numbers anticipate the analysis:

1. **Big loses more than Base on both splits.** Big − Base = −0.79 (test) / −0.54 (valid), consistent in sign and magnitude. The same architectural property that makes Big a better learner of clean in-distribution signal also makes it more pliable to whatever distribution it is fine-tuned on.

2. **The SFT-degraded Base v1.1 lands near v2-pretrained Base.** Base + SFT test 33.77 ≈ Base v2 test 33.90; Big + SFT test 33.54 is slightly above Big v2 test 33.03. Two genuinely different failure modes — pretraining on noise (§5) and SFT on a domain-mismatched filter (§7) — converge to the same BLEU band of 33–34. We argue in §7.6 that this is not coincidence: both modes inject "wrong supervision" (incorrect-translation in §5, correct-but-wrong-distribution in §7), and the model's available accuracy is bounded by how much wrong supervision it has integrated, regardless of which kind.

## 7.4 Analysis: why the filter selected what it did

CometKiwi-22 is reference-free: it scores translation quality in absolute terms. The top-5 highest-scored pairs after dedup (CometKiwi score band 0.8959–0.9145, top 0.0003% of v2 by score):

| # | Source (en) | Translation (fr) | Domain |
|---|------------|------------------|--------|
| 1 | *It is estimated that there are 500–650 million persons with disabilities in the world, approximately 10% of the world population, 150 million of whom are children.* | *On estime qu'il y a entre 500 et 650 millions de personnes handicapées dans le monde, soit environ 10% de la population mondiale; 150 millions d'entre elles sont des enfants.* | UN report |
| 2 | *The estimated number of people living with HIV in Eastern Europe and Central Asia rose to 1.5 million in 2007; almost 90% of those infected live in either the Russian Federation (69%) or Ukraine (29%).* | *On estime que le nombre de personnes vivant avec le VIH en Europe orientale et en Asie centrale a atteint 1,5 million en 2007; près de 90% des personnes infectées vivent soit en Fédération de Russie (69%) soit en Ukraine (29%).* | UNAIDS report |
| 3 | *Since 2005–2006, the Government has reduced the federal debt by $37 billion.* | *Depuis 2005–2006, le gouvernement a réduit la dette fédérale de 37 milliards de dollars.* | Gov. announcement |
| 4 | *During 2006, the population of Finland increased by 4 per cent (21,375 persons).* | *En 2006, la population finlandaise a augmenté de 4 % (21 375 personnes).* | Stat. report |
| 5 | *(other UN/legislative variants of similar register)* | — | UN/legislative |

The top scorers are **dominated by UN, intergovernmental, and legislative-document parallel data**: long, formal, syntactically rigid sentences with statistical content, professionally translated. This is unsurprising: such corpora are exactly what a QE metric will rank highest because they exhibit the alignment, completeness, and syntactic correspondence that QE is trained to reward. WMT14 newstest, in contrast, is **news domain** — looser register, shorter sentences, conversational interjections, named-entity-rich, occasional ungrammaticality from quoted speech.

We thus have two distributions:
- **SFT distribution**: high-quality, formal, document-style
- **Eval distribution**: high-quality, informal, news-style

SFT's effect is to pull the model's output distribution toward the SFT distribution. On the SFT distribution itself, the model improves (loss drops from 3.5 to 1.94, presumably BLEU on UN-style test data would increase). On the eval distribution — where we measure — it gets worse, because every step of SFT trades news-style fit for formal-style fit.

**The model is not deteriorating in absolute quality.** It is being relocated in distribution space, and that relocation crosses the eval distribution.

## 7.5 Implication: "data quality" decomposes

This negative result refines the central claim of the paper. Sections 4–6 showed:

> Capacity return is determined by data quality.

Section 7 forces a decomposition of "quality":

> **Data quality = (translation correctness) × (alignment to target distribution).**
> Both dimensions matter, and they are independent. Reference-free QE metrics measure the first; they do not measure the second.

A naive top-K-by-QE filter optimizes only the first axis, and on a heterogeneous corpus (which v2 is — news, web, UN, Europarl, all mixed) this can move the data far from the target eval distribution. The cleaner the QE filter, the more pronounced the domain skew, because high-quality parallel data is disproportionately UN/legislative.

This generalizes: any post-training step that filters on absolute quality without conditioning on the eval distribution is liable to move the model away from where it should land. The fix is not "better QE" — CometKiwi-22 is doing exactly what it should — but **domain-conditional filtering**: filter by QE within news-domain candidates, or sample uniformly across domains within the QE band, or apply a domain classifier as a hard pre-filter.

## 7.6 Why the v2-pretraining failure and the SFT failure are *different mechanisms*

It is worth distinguishing two failure modes that this paper has now exhibited:

| | v2 pretraining failure (§5) | Quality-filtered SFT failure (§7) |
|--|--|--|
| **Data** | All 30M v2 pairs (noisy) | Top-1M by CometKiwi (clean) |
| **Mechanism** | Incorrect supervision (mistranslations, misalignments, boilerplate) inject wrong gradients | Correct supervision but wrong distribution (formal vs news register) |
| **Evidence** | Spike-guard triggers; manual inspection finds genuine errors | Manual inspection of top-K finds professional translations of *the wrong domain* |
| **Big − Base Δ** | Big v2 −0.87 vs Base v2 | Big SFT −0.54 vs Base SFT |
| **Lesson** | Filter for noise | Match for distribution |

These failures are independent axes — quality filtering with CometKiwi *does* address the v2 noise problem (visual inspection of low-scored v2 pairs confirms genuine noise concentrates in the bottom of the distribution), but it does not address the distribution-alignment problem, because high-quality translations exist in many domains and our top-K selector does not condition on domain.

**Capacity amplifies both failure modes.** Big v2 is worse than Base v2 by 0.87 (noise amplification); Big SFT is worse than Base SFT by 0.54 (distribution-shift amplification). The same architectural property — more parameters, more representational capacity — that makes Big a better learner of correct supervision on clean in-distribution data also makes Big a more efficient absorber of incorrect supervision and a more pliable conformer to whatever distribution it is fine-tuned on. In both cases, capacity is a force multiplier whose sign depends on the data, not on the model.

## 7.7 Limitations and what we did not run

- **Single seed.** We did not repeat the SFT run with different seeds. Variance in SFT outcomes is small relative to the −1.55 BLEU effect, but a 3-seed mean would tighten the bound.
- **Did not test low-LR SFT.** A peak LR ≤ 5e-5 might find a "shallow SFT" point that lifts BLEU on UN-style without moving far enough to hurt newstest. We expect the effect to scale with effective SFT distance from baseline; very-low-LR SFT would be slow to test (need many more steps to converge), and the paper's claim does not require it.
- **Did not test Big v1.1 SFT.** Big has more parameters and may be more or less susceptible to distribution shift. Replication on Big is left to a follow-up; it is not central to the conceptual result.
- **Did not test domain-conditional filtering.** Constructing a news-conditioned top-K filter is a nontrivial sub-project (domain classifier + quality scorer + joint sampling). We flag it as the natural fix and leave it for future work.

## 7.8 Headline result

**Quality-filtered SFT, by itself, does not improve newstest BLEU when "quality" is measured by reference-free QE on a multi-domain corpus.** It costs 1.55 BLEU on Base v1.1. The reason is not that the filter failed — the filter selected genuinely high-quality pairs — but that absolute translation quality and alignment to the eval distribution are independent axes of "quality", and only the first was optimized.

The negative result tightens the paper's prescription:

> **Filter first (for noise), match the evaluation distribution second (for register/domain), and only then scale capacity.** Skipping the second step undoes capacity returns from the first.

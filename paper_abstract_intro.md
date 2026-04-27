# Title (working)

**When Capacity Pays: A Single-GPU Replication of Vaswani 2017 with a Two-by-Two Data-Quality Ablation**

Alternative titles considered:
- *Capacity Returns Are Conditioned on Data Quality: A Reproducible Single-GPU Study of WMT14 Transformers*
- *Filter First, Scale Second: When Big Beats Base in Transformer Machine Translation (and When It Doesn't)*

---

## Abstract

We report a from-scratch single-GPU (RTX 5090, 32GB) replication of the Transformer (Vaswani et al., 2017) on WMT14 en-fr, en-de, and WMT17 zh-en, and use it as a controlled testbed for the relationship between data quality, model capacity, and post-training. We obtain test BLEU 35.31 (Base, 60M) and 35.87 (Big, 209M) on en-fr newstest2014 and 24.04 (Base) on en-de newstest2014, all averaged checkpoints with sacrebleu 13a — within or above the paper's sacrebleu-equivalent band. To probe the data-quality / capacity interaction we run a clean two-by-two: {strict-filter 9.3M, loose-filter 30M} × {Base, Big}. **The capacity return changes sign with data quality**: on clean data Big exceeds Base by +0.56 BLEU; on noisy data Big falls below Base by −0.87. The same 3.5× extra parameters thus pay or cost depending only on the corpus they see. As a post-training addendum, we test whether quality-filtered SFT (top-1M of the noisy 30M by CometKiwi-22) recovers the lost ground; it does not. SFT degrades both Base and Big by 1.5–2.3 BLEU on newstest, with Big degrading more, because the QE-top of a multi-domain corpus is heavily UN/legislative — high-quality but the wrong distribution for news. We argue this decomposes "data quality" into two independent axes — translation correctness × alignment to the eval distribution — and reference-free QE measures only the first. Finally, we offer a methodology note on loss-spike-guard trigger rates, which we initially treated as a noise meter but find to be a joint function of data noise and model convergence depth (Big on clean data triggers more often than Big on noisy data). All training code, configurations, four checkpoints, and stdout logs are released; the entire study reproduces in approximately 30 GPU-hours on a single 5090.

---

## 1. Introduction

The Transformer (Vaswani et al., 2017) is the most-cited baseline in machine translation, and its WMT14 numbers — 27.3 tokenized BLEU for Base on en-de, 38.1 for Big on en-fr — are the reference point against which post-2017 progress is measured. Reproducing those numbers is harder than the paper's compactness suggests: their tokenized BLEU is not directly comparable to modern sacrebleu, the data preprocessing pipeline is under-specified, and the original training runs used 8-GPU configurations that no longer match the typical reproduction environment of a single accelerator. Most work that cites Vaswani 2017 quotes the numbers without re-running them. This paper does re-run them, on a single RTX 5090, and uses the resulting infrastructure to ask a sharper question.

**The question.** Modern empirical ML has converged on two largely-independent levers for improving a learning system: *more capacity* (Kaplan et al., 2020; Hoffmann et al., 2022) and *more, cleaner data* (Penedo et al., 2023; Gururangan et al., 2020). It is widely assumed that pulling either lever, holding the other roughly fixed, produces monotone gains. Our study contradicts that assumption in a sharp form. On WMT14 en-fr, scaling Base (60M) to Big (209M) **changes sign as a function of data quality**: on a strictly-filtered 9.3M corpus, Big beats Base by +0.56 BLEU; on a loosely-filtered 30M corpus drawn from the same source pool, Big falls below Base by −0.87 BLEU. The same architectural intervention pays in one regime and costs in another; the regime is determined entirely by data quality.

**What we do.** We construct a clean two-by-two design: data quality (strict-filter v1, loose-filter v2) × model capacity (Base, Big), and report all four trained-from-scratch results on the same single-GPU pipeline. We extend the design with cross-language replication (WMT14 en-de, where the same Base > Big ordering holds at 4.5M scale) and a post-training intervention (SFT on the top-1M of the noisy 30M, scored by CometKiwi-22). The SFT result is a clean negative finding that tightens the central claim: data quality is not a single number — it decomposes into translation correctness *and* alignment to the eval distribution, and naive top-K-by-QE filtering optimizes only the first.

**Contributions.**

1. **A reproducible single-GPU Transformer pipeline** that hits 35.31 / 35.87 sacrebleu on WMT14 en-fr Base/Big and 24.04 on en-de Base, with full open-source code, configurations, four published HuggingFace checkpoints, and persistent training logs.
2. **A 2×2 ablation establishing that capacity return is sign-conditioned on data quality** (+0.56 vs −0.87 on the same en-fr pool, replicated qualitatively on en-de).
3. **A post-training negative result** showing that quality-filtered SFT, at scale, degrades newstest BLEU by 1.5–2.3 and that the failure is mechanistically distinct from §5's pretraining-on-noise failure but converges to the same BLEU ceiling.
4. **A methodology note** on loss-spike-guard trigger rates as a diagnostic, with evidence that they cannot be interpreted as a pure data-noise metric.

**Scope and limitations.** We run single seeds per configuration. The variance of any one BLEU number we report is unmeasured, but the +0.56 / −0.87 / −1.55 / −2.33 effects we make claims on are large relative to the typical Transformer-MT seed-variance band of ~0.2–0.4 BLEU, and our cross-language and within-paper replication patterns argue for the directional findings. We use sacrebleu 13a throughout; we do not run human evaluation. We test only en-fr, en-de, and zh-en (Sections 3, 4–6, 7); generalization to morphologically richer pairs (e.g., en-cs, en-ru) is not in scope.

**Roadmap.** §2 reviews related work. §3 describes the single-GPU setup and reports the zh-en mode-collapse failure (the project's first run, included to document a real reproduction risk). §4–§5 give the en-fr v1 success and v2 failure runs that constitute the data-quality axis. §6 generalizes the pattern to en-de. §7 reports the quality-filtered-SFT negative result and the multi-dimensional view of "data quality" it implies. §8 is the methodology note on spike-guard interpretation. §9 concludes.

# EAMT 2027 paper outline (draft, 2026-10-16)

Skeleton of the EAMT 2027 research-technical paper. It is written for the author, so it uses the
project's internal names (E0.5, `ft_un_div`, ...). The paper itself must not use them: see 0.2.

**Sources used, and nothing else** (every number below comes from one of these):

- `phase0/README.md`, "E0.5 RESULT (2026-10-16)", "E0.4 RESULT", "E0.3 RESULT" and its lr-1.0 follow-up, "Rebuild results", "QE scoring inventory".
- `PROTOCOL.md`, "E0.4", "E0.5", "E0.3 decisions" (D1-D9).
- `phase0/PRIOR_ART_E04.md`.
- `phase0/next/groundwork.json`: `notes` (10 papers read in full), `llm`, `ende`, `vocab` memos.
- `phase0/next/e06_feasibility.md`. `vocab/` and `ende/` below mean `phase0/next/vocab/` and
  `phase0/next/ende/`; their bulk files are in `~/mt_local/phase0_next/` (sha256 in `phase0/next/SHA256SUMS`).
- The previous `WMT2027_PLAN.md` (in git history; what survives, its section 8) and the new
  `WMT2027_PLAN.md` that replaced it (schedule, budget, the W4 placement of remedy arms, the reading deadlines in its §6.2).
- `phase0/prereg_drafts.md` ("prereg" below): the only place where the E0.6, E0.7
  and E1.0 outcome rules, the reading of each outcome and the OPEN DECISIONS ("OD") are stated.
  This outline says what the paper puts in each slot and points to prereg for how an outcome is
  read; it does not restate those readings.
- Scratch counts made for this outline, read only, from `~/mt_local/phase0_final/e05_src/` and
  `e05_results/data/phase0_e05/` (UN share of `ft_topk`; top-ranked share of the random sets; the
  `rep_dedup` / `div_dd` rebuild); to be re-derived by the report and build scripts.

File names in section 4 were read from `~/mt_local/` (read only). "To check" marks anything not
verified. "Scratch" marks numbers computed by the groundwork agents with scratch code; they must be
rebuilt by a pinned, fail-closed script before they go into the paper.

---

## 0. Ground rules for this paper

### 0.1 Venue facts

| item | value | status |
| --- | --- | --- |
| Target | EAMT 2027, research-technical track | fixed |
| Length | up to 10 pages | fixed; whether references, limitations and appendices count is **to check** when the 2027 call is out |
| Deadline | 2027 call not out. EAMT 2026: 18 March 2026, extended to 27 March | planning assumption: mid-March 2027; the 2026 dates are from a web lookup during planning, not in the sources above, **to check** |
| Acceptance | EAMT 2026 research-technical: 31 of 55 (56%) | web lookup during planning, not in the sources above; **to check** |
| Review mode, template, anonymity of pre-registration links, supplementary material | unknown | **to check** |
| Follow-up | WMT 2027 (summer 2027), extended version | fixed |
| arXiv | not used (V1 was moderation-rejected; peer-review path only) | fixed |

### 0.2 Terminology: internal name -> what the paper says

| internal | in the paper |
| --- | --- |
| top band, `ft_topk` | "the top-ranked set": the 1,000,000 highest-scoring pairs, 926,999 after removing exact duplicate pairs |
| `ft_random`, `ft_bottom` | "a random set of the same size", "the lowest-ranked set" |
| `rep`, rep > 0 / rep = 0 | "recurrence": the share of a sentence's source 8-grams that occur more than once in the top-ranked set; "repetitive" / "non-repetitive" half |
| `ft_un_div`, `ft_un_rep`, `ft_un_random` | UN-only non-repetitive half, repetitive half, random UN set |
| `heldout_un` | "held-out UN test set" (2,000 pairs not in the base model's training data) |
| lr_scale 1.0 / 0.15 | the base model's learning-rate schedule continued at full scale / at 0.15 of it (exact wording **to check** against the trainer) |
| seed-noise floor | "the larger of the two seed standard deviations" |
| pretraining exposure | "share of pairs already in the base model's training data" |
| E0.3, E0.4, E0.5 ... | never named; describe the comparison |
| "from scratch" (groundwork's name for the NMT arm) | **never** use it for our arm. Our NMT runs fine-tune a model we trained ourselves; "from scratch" in this literature means training on the selected data from random initialisation (Waldendorf, Chimoto, Finkelstein, Peter). Call it "the NMT arm" or "the Transformer arm". E0.6 is a fine-tuning experiment with E0.5's recipe. |

### 0.3 Sentences the paper must not contain

- "first to observe / first to notice" anything. The phenomenon is in Waldendorf 2025, Chimoto 2024, Tan 2024, Finkelstein 2024; the redundancy remark is in Lo 2018 and Bernier-Colborne & Lo 2019.
- "QE metrics select repetitive text" as a general property (one metric, one pool, one pair today).
- "repetitiveness explains x% of the gap" (E0.5 registered that no claim is drawn from the size of the UN gap against the all-source gap).
- "predicts" (a median split is a two-point contrast).
- "the effect is domain shift" (contradicted: the top-ranked set lowers in-domain BLEU).
- "QE filtering hurts" without scope (Peter 2023 shows the opposite at 50% kept, trained to convergence from scratch).
- "a diversity control fixes QE selection" (no remedy result exists).
- "equal" or "no difference" for a contrast that was only not detected; say "not detected" and give the smallest detectable gap.

---

## 1. Working titles

1. **What Does CometKiwi Select? A Controlled Study of Repetition in Top-Ranked Fine-Tuning Data** (recommended: names the one QE metric tested and keeps the fine-tuning scope; rule 0.3, ledger 20).
2. **Why Do CometKiwi's Highest-Scoring Pairs Fine-Tune Poorly at a High Learning Rate? Repetition in Top-Ranked Fine-Tuning Data** (question form; the harm must be scoped in the title: it holds only at lr_scale 1.0, and at 0.15 no top-vs-random difference was detected).
3. **Repetition at the Top of CometKiwi's Ranking, and What It Costs MT Fine-Tuning** (use only if E0.7 or E1.0 replicates, since it reads as a general statement across pairs or models; both still select with CometKiwi-22 alone, so the title must name it; rule 0.3, ledger 20, Limitations 1).

---

## 2. Draft abstract (about 230 words with the slots; trim if the 2027 call caps abstract length)

> Reference-free quality estimation (QE) is often used to select translation training data, yet
> several studies report that the highest-scoring pairs train no better, or worse, than a same-size
> random draw, a lower-scoring band or the full data. We ask
> why. On WMT14 English-French, we fine-tune a trained Transformer on the pairs CometKiwi-22 ranks
> highest, split by whether their source sentences share 8-word sequences with other selected
> sentences. With source corpus, size and length matched, and QE score and pretraining exposure
> nearly equal between the halves, the repetitive half costs 2.63 BLEU more on out-of-domain news
> than the non-repetitive half (five seeds, pre-registered). That half also has fewer source word
> types and 8.45% duplicate source sentences, which this test does not separate from repetition.
> The non-repetitive half is still 0.58 BLEU below random text from the same corpus (the random
> text is not QE-matched). [E0.6: against a non-repetitive set matched to it in vocabulary coverage, the
> gap to the repetitive half (duplicates still present) is {X} BLEU; without exact duplicates,
> against a non-repetitive set not matched on vocabulary, {Y}.] [E0.7: On English-German, {Z} BLEU.] [E1.0: For a 1.7B LLM, {W}
> MetricX.] The English-French split contrasts use a learning rate at which even random data
> lowers news BLEU; at a conservative rate,
> no difference between top-ranked and random data was detected (three seeds; the smallest
> difference that test could have detected was 0.43 BLEU).

Traceability: 2.63 and 0.58 are E0.5 Q1 and Q2 (README). "QE score nearly equal" is 0.8990 vs
0.8993 for the two halves (vocab memo, scratch; re-derive); the random UN set is 0.8629, so the
0.58 contrast is not QE-matched. QE and exposure were measured, not matched by design. "Pretraining
exposure nearly equal" is 36.7% vs 36.7% as exact pairs (PROTOCOL E0.5 table, a dry run on the Mac;
re-derive from the run-of-record sets; reproduced exactly on 18 Oct, 36.73 / 36.71%); as source
sentences it is 36.8% vs 38.3% (vocab memo, scratch), so the paper must name the measure. It is equal
as a share of rows; how often the base saw those pairs is not equal: the repetitive half's exposed pairs sit on more corpus lines (1.158 against 1.043), so per row the base presented its pairs 1.056 times against 0.951 (base-replay check, 18 October 2026); say so beside the share. "Fewer source word types and 8.45% duplicate
source sentences": 25,134 vs 33,874 types and 8.45% vs 0% (README, E0.5, "What it does not establish"; the
strands are not separated there). The E0.6 slot is worded neutrally because E0.6 has not run, and
it names what each contrast leaves unmatched: the coverage-matched contrast (prereg E0.6 OD 1's
measure) still carries the duplicates (5.1) and the duplicate-free contrast is not
coverage-matched (5.2). A contrast that is
not detected fills its slot with "not detected" and its smallest detectable gap (rule 0.3). "The English-French split
contrasts": the E0.5 contrasts and the E0.6 slot (E0.5's recipe, lr_scale 1.0). Not "all
English-French Transformer contrasts": the conservative-rate contrast (E0.3, lr_scale 0.15) is one
too, and there random data did not lower news BLEU (35.43 vs the 35.31 baseline, +0.12; README, E0.3
RESULT). E0.7 (en-de) has no random-vs-base result and E1.0 picks its own LoRA rate on FLORES+
dev, so the sentence covers neither. "Even random data lowers news BLEU": lr-1.0 random
34.41 vs baseline 35.31 (README), and E0.5's random sets -1.83 / -4.14 (README, E0.5); the 0.90 loss is larger than the ~0.2-0.4 BLEU by which the
averaged baseline biases every "vs base" change (3.3, Evaluation). "No difference detected at a
conservative rate": E0.3 at lr_scale 0.15, gap 0.26, t = 2.59, df 2.6, critical 4.30 at the
floor(df) lookup (README); smallest detectable gap 0.43 BLEU with these 3 seeds (0.34 under E0.5's
exact rule; derived after the fact with `probe_stats.contrast` from `phase0_bleu.tsv`, since E0.3
did not register this bound; re-derive in the report script). If a slot's experiment fails or is not run, delete the slot; do not soften it
into a claim.

---

## 3. Section by section

Each section: purpose, evidence in hand, gap, experiment that fills it.

### 3.1 Introduction (about 1.0 page)

**Purpose.** State the problem; credit the known phenomenon before stating our question; give the
scope in the first page so Peter et al. 2023 cannot be cited against an unscoped claim.

**Paragraph plan.**

1. QE-based selection is common practice. At the extreme top it can fail: credit Waldendorf 2025
   (CometKiwi-22 top 1% vs same-size random), Chimoto 2024, Tan 2024, Finkelstein 2024. NRC noted in
   2018 and 2019 that top-ranked sentences are redundant (Lo 2018; Bernier-Colborne & Lo 2019). None
   runs a controlled test of why (Lo 2018 tested a redundancy filter, single runs; Chimoto,
   Waldendorf and Finkelstein offer explanations without testing them).
2. Our question: what does the top of a CometKiwi-22 ranking contain, and which of its properties
   costs translation quality when a trained model is fine-tuned on it?
3. Approach: matched splits of the top-ranked set (size, length histogram, source corpus), random
   controls matched on size and length (not on QE), five seeds, pre-registered decision rules, token accounting.
4. Findings in hand: 2.63 / 0.58 (E0.5); in-domain pattern (2.1-2.7 BLEU lost by both halves, +0.20
   for random UN). Slots for E0.6, E0.7, E1.0.
5. Scope sentence (must stay): one language pair (English-French), one base model, the top 6.0% of
   a 16.6M-pair pool, fine-tuning an already trained model, at a learning rate where random data
   also costs news BLEU; one QE metric.

**Contributions (draft wording; keep to three).**

1. A measurement of what CometKiwi-22's top-ranked set contains against a same-size random draw
   (vocabulary, duplicate rate, 8-gram recurrence, length) on a curated pool. The random draw is not
   length-matched (p90 length 35 vs 47); a whole-set vs length-matched random comparison at
   identical word counts does not exist for en-fr (CPU to compute; the ende memo did it for en-de).
2. A controlled, pre-registered test: inside one source corpus, at matched size and length, the
   repetitive half costs 2.63 BLEU more out of domain; the non-repetitive half still trails random
   text by 0.58. [E0.6: the repetitive half against a coverage-matched set and, without exact
   duplicates, against a non-repetitive set; a vocabulary-capped random set vs the rest of
   selection.]
3. [E0.7 / E1.0: the same contrasts on English-German and on a 1.7B LLM.]

Do not list "multi-seed" or "pre-registration" as contributions; they are methods.

**Gap.** Contribution 3 does not exist until E0.7 and E1.0 report. If neither reports in time, the
paper is a one-pair, one-model study and the title must be option 1 or 2.

---

### 3.2 Related Work (about 1.2 pages)

**Purpose.** Credit the phenomenon and the redundancy remark; name the competing accounts (coverage,
pool noise, metric training coverage, selection fraction); state where each paper differs from our
setting. All papers below were opened (PRIOR_ART_E04.md checkers, the groundwork notes or the llm
memo); Biderman et al. abstract only. The author must still open each before it enters the .bib.

**Subsections and positioning sentences.**

#### (a) QE selection and its top slice

| paper | positioning sentence | check before citing |
| --- | --- | --- |
| Peter et al. 2023, WMT (2023.wmt-1.50) | Keeping the QE-best half of WMT data and training from scratch to convergence beats a random half in every COMET22 cell (six directions) and full data on most (e.g. en-de WMT'23: 83.5 BleurtQE, 82.9 CometKiwi, 81.8 full, 80.8 random; full data scores higher on IWSLT'23 for ja-en, 85.5 vs 85.0/85.1, for zh-en, 84.9 vs CometKiwi 84.5 (BleurtQE not in the note, to check), and by 0.1 for en-zh and de-en); we study a different part of the ranking (top 6%) in a different regime (fine-tuning), and the two results can both hold. | CometKiwi checkpoint not named; seeds not stated; the 50% choice rests on unreported preliminary runs; some appendix BLEU cells put a QE half below random (not discussed by them; do not lean on them) |
| Waldendorf et al. 2025, Findings of EMNLP | CometKiwi-22's top 1% is no better than a same-size random draw for En-De on chrF++/spBLEU (lower COMET, 0.763 vs 0.783) and clearly worse for En-Ar (45.2 vs 49.7 chrF++), from scratch, 3 seeds; they note in passing (App. C) that their own LLM-label filter (MDFS, not CometKiwi) selected many similar sentences in early En-Ar runs, and near-deduplicate the pool; we test that property inside the top-ranked set in fine-tuning, as the bundle our split selects (shared 8-grams, fewer word types, exact duplicates). | checkpoint not named; whether their dedup covers En-De; their explanation (metric training coverage) must be addressed |
| Chimoto et al. 2024, Findings of ACL | Keeping the top 10% of a WMT19 en-de sample by COMET-QE falls far below random (16.7 vs 23.7 FLORES; 19.4 vs 28.6 WMT18) while 50% kept is slightly ahead; they blame pool noise and never inspect the selected subset. | paper never states which end is kept; COMET-QE 2020, not CometKiwi; the "+/-" values are unlabelled and appear to be paired-bootstrap intervals (note), not seed spread |
| Tan et al. 2024, WMT (system paper) | For LLaMA2-7B en-zh fine-tuning, 53k pairs above 89 give 33.5 BLEU against 36.1 for 53k above 87, and 100k above 87 match 100k sampled pairs (35.5 vs 35.7); single runs, no explanation. | "100k Cleaned" is described two ways (random vs low-QE); whether the >87 set includes the >89 pairs; filtering checkpoint unstated |
| Finkelstein et al. 2024, WMT | Training from scratch on the top ~0.3% of WMT'23 en-de by BLEURT-QE gives COMET22 43.18 against 78.79 for the full data; no random control, no analysis of the slice; they blame the fraction. | no BLEU anywhere; filtered set is sentence-level only while test sets are paragraph-level (their own confound) |
| Fujita de Mello & Takada 2025, IEEE BigData (arXiv 2512.11388) | For 7B LLMs with LoRA on KFTT ja-en, CometKiwi top-10k beats random in 8 of 8 BLEU cells, but the pool was deduplicated and repetition-filtered before scoring, runs are single-seed and unmatched in length; this is the closest counter-result for E1.0. | tables contain apparent copy errors (Gemma and Mistral cells identical); test-set provenance unclear; venue only from arXiv comment |
| Koehn et al. 2018, WMT18 filtering findings | The WMT18 task fixed the protocol of keeping the top-scored 1% or 10% (English words) of a deduplicated noisy pool and training from scratch, with no random baseline; we add random controls matched on size and length and look inside the selected set. | pool size in pairs not stated; the NMT architecture is not named (Marian v1.5 defaults); RNN is inferred from "dropout for RNN states", so do not call it a Transformer |
| Koehn et al. 2019, WMT19 filtering findings | Several submissions already built vocabulary coverage into selection; the findings do not test whether it helped. | DiDi not in the coverage sentence |

#### (b) Redundancy in top-ranked data, and redundancy controls

| paper | positioning sentence | check before citing |
| --- | --- | --- |
| Lo et al. 2018, WMT18 (NRC) | Reported redundancy and limited vocabulary among the top-ranked pairs of an MT-plus-YiSi-1 scorer on noisy ParaCrawl, and a filter requiring a new source bigram per kept pair (+2.44 BLEU NMT at 10M words internally, +0.91 in the official six-domain evaluation, little at 100M); single runs, not a QE metric, from scratch; we measure the property for CometKiwi-22 and test it under matched controls. | "virtually no effect" at 100M is +0.73 dev / +0.71 test for YiSi-1 and +0.45 / +0.14 for HMM in their Table 2; cite numbers, not wording; refill rule for dropped pairs unstated |
| Bernier-Colborne & Lo 2019, WMT19 (NRC) | Observed redundancy in the top ranks of YiSi-2 and applied a 20% discount to pairs with no new source bigram, never ablated. | YiSi-2 is embedding similarity, not a trained QE model; their "random" baseline also had the re-ranking |
| Kirchhoff & Bilmes 2014, EMNLP | Per-sentence top-N selection cannot represent redundancy; our setting is the QE-specific instance. | none |
| Azpeitia et al. 2018, WMT18 (Vicomtech) | Source n-gram saturation on top of a ranking hurt NMT (30.40 to 24.91 average BLEU at 100M words; 26.35 to 25.64 at 10M), so a naive redundancy control is known to hurt in one setting. | none |
| Lu et al. 2018, WMT18 (Alibaba) | Feature-decay diversity re-scoring after a quality sort: NMT 25.01 to 25.34 at 10M words, no change at 100M. | motivated by domain coverage, not scorer bias |
| Rossenbach et al. 2018, WMT18 (RWTH) | A near-duplicate filter gives +3.2 to +3.9 BLEU on random samples but 0.0 to +0.4 once pairs are model-ranked. | does not show a ranked top set needs the control |
| Xu et al. 2020, WMT20 (Volctrans) | An n-gram coverage discount over a score-sorted list adds +0.03 to +0.19 BLEU (single runs). | km-en gains confounded with a language-ID discount |
| Wang et al. 2018, WMT18 (NICT) | Remarked, unmeasured, that very short sources may score very high and give a short, redundant selection; our split is length-matched, so our effect is what that remark does not cover. | cite Anthology values W18-6489, pp. 963-967 |
| Fernando et al. 2025, EMNLP | Top-ranked pairs under embedding-similarity scorers are biased (short, number/URL overlap), and 4- to 7-gram deduplication helps NMT; no QE metric. | re-check figures against the PDF; spell "Velayuthan" |

#### (c) Forgetting during fine-tuning, and vocabulary coverage

| paper | positioning sentence | check before citing |
| --- | --- | --- |
| Saunders & DeNeefe 2024, EMNLP | Generic-quality loss after domain fine-tuning tracks the fine-tuning set's vocabulary coverage (Kendall tau 0.9 for target coverage vs COMET drop, 5 de-en and 3 en-ja domains, single runs), and a small coverage-completing mix-in removes most of it, though a random 1:1 mix-in (10-50x larger) reduces the drop at least as much (de-en mean BLEU drop 1.5 vs 2.2 for Minimal Mix-in; Tables 8-9), so the paper does not separate a coverage-specific benefit (this weakens the coverage account 5.1 might adopt); we test, inside one domain and with vocabulary coverage matched on one measure (prereg E0.6 OD 1), whether the repetitive half (recurrence with its duplicates) costs quality beyond that measure [say "beyond their coverage" only if OD 1 picks their measure]. [Keep the second clause only if prereg E0.6's C1-beyond-coverage outcome is YES; for any other outcome, use the reading prereg states for it (5.1).] | n = 8 pooled or per pair; no same-size random control for their mix-in; their coverage is subword types of the generic vocabulary, ours is word types; compute theirs before claiming agreement |

#### (d) What QE metrics reward, and metric interference

| paper | positioning sentence | check before citing |
| --- | --- | --- |
| Sun et al. 2020, ACL; Agrawal et al. 2024, EMNLP | QE models can score well from one side and pick up lexical artifacts, and CometKiwi-22 separates high-quality translations only weakly; so the top plateau (0.8959-0.9145) may be ordered by properties other than quality. | not evidence for repetitiveness |
| Zhang et al., arXiv 2510.22028 (PDF header: COLM 2026) | QE metrics including CometKiwi score longer error-free passages lower; this is why our splits are length-matched. | venue only from PDF header; do not cite as showing top-k keeps short sentences |
| Pombal et al. 2025, arXiv 2503.08327 | Metrics favour systems trained on data selected by the same or related metrics; this is why E1.0's primary metric is MetricX and COMET-22 is flagged; they also found CometKiwi top-k beating random for Gemma-2-2B (en-zh COMET 79.28 vs 76.78), a counterpoint for E1.0. | read via the LLM memo; author to read |

#### (e) Data diversity beyond NMT filtering (one paragraph)

Pan et al. 2024 (G-DIG; diversity benefit significant only at 2,000 examples or fewer); Bukharin et
al. 2024; Wettig et al. 2024 (QuRating: pure top-k by quality rating worse than temperature sampling,
perplexity 10.59 vs 8.91); Zheng et al. 2023 (top-of-score below random at high pruning, but their
top set is hard examples, the opposite mechanism); Dankers et al. 2023 (the "easy" region of NMT
data has the lowest unique-trigram ratio, preliminary appendix result). One sentence each; none
involves a QE metric on MT data.

#### (f) LLM fine-tuning for MT (only if E1.0 reports)

ALMA (Xu et al., ICLR 2024: too much parallel data washes out an LLM's prior; COMET peaks at
10K-100K pairs for LLaMA-2-7B en-ru), Biderman et al. 2024 (LoRA forgets less; abstract only, author
to read), Shang et al. (arXiv 2602.11961; the 8-shot reference numbers for the pilot gate).

**Gap.** Not opened by anyone: the WMT19 filtering system papers behind Koehn et al. 2019, i.e.
Erdmann & Gwinnup 2019 (AFRL), Gonzalez-Rubio 2019 (Webinterpret) and Axelrod 2019 (DiDi, cynical
selection), which PRIOR_ART section 5 flags as where the real pre-emption risk for its C3 sits
(that QE selection needs a diversity or redundancy control; that a repetitiveness statistic
predicts the harm); Erdmann & Gwinnup 2018 (AFRL, "Coverage and cynicism"; Koehn 2018 note); the
NRC unsupervised submission (Littell et al. 2018; Lo 2018 note); "The repetition rate of text as a
predictor of the effectiveness of machine translation adaptation" (PRIOR_ART Appendix B; authors,
venue and year **to check**; it also bears on the "predicts" ban, ledger 23); Karpinska et al. 2022
(metric sensitivity to repetitions; PRIOR_ART section 5); and, as an E1.0 / forgetting lead, Pang et
al. 2024 (Saunders note). When each must be read, and before which sentence of this paper, is
WMT2027_PLAN section 6.2 ("Read before a design or claim depends on them"); no Related Work sentence
about redundancy or a coverage control is written before then.

---

### 3.3 Setup (about 1.4 pages)

**Purpose.** Everything a reviewer needs to rebuild the sets and judge the contrasts; the learning-rate
regime stated here, not hidden in Limitations.

**Evidence in hand (all README / PROTOCOL).**

- **Corpus.** WMT14 en-fr, rebuilt with a carriage-return-safe pipeline: 38,275,284 pairs; the
  rebuild's legacy mode reproduces the published corpora byte for byte (one sentence in the body; the
  pipeline story goes to an appendix, without citing the rejected paper).
- **Base model.** The project's Transformer Base trained on the v1.1 corpus (9,312,233 pairs),
  averaged checkpoint at global step 105,000: newstest2014 35.31, held-out UN 46.83, newstest2013
  30.52. 60.5M parameters (strict load of the release; README, "Findings from the original training
  machine", item 4). The base is a public release; whether naming it breaks anonymity is **to check**.
- **Pool.** The 16,646,992 pairs whose CometKiwi score carried over from the original scoring run
  (E0.3 decision D1, "reused"): UN 11,598,490, Common Crawl 3,055,442, Europarl 1,930,749,
  news-commentary 26,268, giga-fren 4,351 (D7 counts; the remaining 31,692 rows are not attributed to
  these five sources, derived as 16,646,992 minus the D7 counts). Suggested body sentence: "the pool
  fixed before any run (D1, frozen 2026-09-18: the 16.6M pairs with original scores); on the full
  38.3M-row pool, 55% of it giga-fren text the base never saw, no learning rate passed the frozen
  rule (secondary analysis, not pre-registered)". Do not call it "the pool the base model saw": inside
  v1.1 are about 49.8% of its top 1M and 56.0% of a random 1M (D1, approx.), and the E0.5 UN sets are
  36.7 / 36.7 / 37.9% exact pairs in pretraining (PROTOCOL dry-run table). Wording that defines this
  pool without the pipeline history is **to check**.
- **QE.** CometKiwi-22 (`Unbabel/wmt22-cometkiwi-da` at pinned commit `1ad78519...`, InfoXLM-large).
  Rescoring reproduces the original scores: mean difference -2.7e-9, p99 |diff| 1.02e-06, Spearman
  1.0000000 over 16,646,992 rows.
- **Top-ranked set.** Top 1,000,000 by score (6.0% of the pool; scores 0.8959-0.9145; UN 78.3%,
  Europarl 15.6%, Common Crawl 5.7%, news-commentary 0.2%; PROTOCOL D1, before deduplication),
  926,999 rows after removing exact duplicate pairs (73,001 removed; 5.6% of the pool): 77.0% UN
  (713,881 rows), Europarl 16.8%, Common Crawl 5.8%, news-commentary 0.2%, unattributed 0.2%
  (counted from the pinned `ft_topk.source`, `a4ad4586...`; the lr-1.0 manifest gives the same
  counts). The trained set is the 926,999 rows, so it is 77.0% UN; 78.3% is the 1M before
  deduplication. Random set (`ft_random`, 926,999 pairs drawn uniformly over unique pairs): UN 66.6%,
  Common Crawl 20.2%, Europarl 12.9% (counted from the pinned `ft_random.source`, `fd675a19...`); a
  1M row draw with duplicates is 69.6% UN (PROTOCOL D1).
- **Recurrence statistic.** Per row, the share of its source 8-grams (lowercased `\w+` tokens) whose
  count over the top-ranked set exceeds one; rows under 8 tokens get 0. Median 0.0, so the split is
  "any recurring 8-gram" vs "none". As run, an 8-gram repeated inside one row also counts: 59 of
  926,999 rows differ from the registered wording, 44 change sides (appendix A).
- **Matching.** Halves: per source-length bucket take min(n_div, n_rep) rows from each side (seed 42),
  so length histograms are identical. Random controls: exactly as many rows per length as the halves,
  drawn at random within the bucket (seed 42); the build fails if any bucket is short (tightest slack
  2.24x all sources, 1.66x UN). Top-ranked rows are not removed from random draws, so they keep
  top-ranked rows at their base rate: 8.8% of `ft_un_random` and 7.3% of `ft_random_small` are
  exact pairs of the top-ranked set (scratch count against `ft_topk`; re-derive in the report
  script). The 2.2-3.1% in PROTOCOL E0.5 is a different quantity: the share of a random set's
  source lines that also occur in each half (manifest `source_lines_also_in_*`), not the share of
  top-ranked rows.
- **Sets** (E0.5): `ft_topk_div`/`ft_topk_rep`/`ft_random_small` 242,022 rows each (UN 74.2 / 90.9 /
  66.4%); `ft_un_div`/`ft_un_rep`/`ft_un_random` 214,489 rows each (100% UN). Table 1.
- **Fine-tuning.** From the averaged base checkpoint, 10,000 micro-steps (README: "the whole
  10K-micro-step SFT run"; D9: every arm stops at 115,000 micro-steps from global step 105,000),
  lr_scale 1.0, micro-batches capped at 192 sentences (so equal steps means equal sentence pairs:
  10,000 x 192 gives 7.9 passes over 242,022 rows, 9.0 over 214,489), loss-spike guard off, final
  checkpoint, seeds 42/1/2/3/4. Exact wording (micro-steps vs optimizer steps) **to check** against the
  trainer config. Optimizer state at the start of fine-tuning (README: Adam state discarded) **to
  check** wording.
- **Evaluation.** newstest2014, beam 5, sacreBLEU with 13a tokenisation (full signature **to check**);
  held-out UN, 2,000 pairs drawn from pairs absent from the base's training data, one over-long row
  dropped from every system alike. Baseline caveat (README, lr-1.0 result): the baseline is the
  averaged release while the lr_scale 1.0 arms use single final checkpoints, which biases every
  change "vs base" toward "degrades" by the ~0.2-0.4 BLEU averaging is worth; condition-vs-condition
  contrasts are between like checkpoints and unaffected. Random's 0.90 news loss survives it.
- **Decision rule.** The gap test (a gap counts if it exceeds the larger seed sd of the two sets and
  is Welch-significant at 0.05) was applied in E0.3 (top vs random), E0.4 and E0.5. E0.3's
  registered gate had more: criterion 1 also required the top-ranked set to degrade news against the
  baseline beyond the floor, and criterion 2 required an in-domain gain above the top-ranked set's
  seed sd (README, "Pre-registered decision rule"; `e03_decide.py`). E0.4 registered "by more than
  the seed-noise floor", applied with the Welch test by `e04_report.py`. Reporting a non-detection
  with the smallest gap the rule could have called was registered only for E0.5 (PROTOCOL E0.5,
  "The rule, for every contrast"); the E0.3 and E0.4 bounds quoted here (0.43 / 0.34; 0.38 / 0.35)
  were computed after the fact. The E0.5 contrasts use the exact critical value at fractional df (new in E0.5,
  `probe_stats.py`); the E0.3, E0.4 and lr-1.0 contrasts used `e03_decide.py`'s floor(df) lookup,
  which is stricter (so their critical values, 4.30 at df 2.6 and 3.18 at df 3.5 / 3.6, are floor(df)
  values). With five seeds the E0.5 rule calls a true 0.5 BLEU gap 67-92% of the time and a 0.3 gap
  30-51%; false alarms 3.4-4.5% (PROTOCOL E0.5's simulation at E0.4's seed sds of 0.20-0.28).
  At E0.5's own UN seed sds (0.11-0.14) the rule could call 0.18-0.19 BLEU, and at a seed sd of
  about 0.13 about 0.26 BLEU has 80% power (vocab memo; prereg E0.6, "Why five seeds").
- **Tokens.** Equal sentence pairs, unequal tokens: Q1 pair 2.15%, Q2 pair 2.42% apart; by the runs'
  own late dev-BLEU slopes worth at most 0.043 and 0.034 BLEU against a floor of 0.14.
- **Learning-rate regime** (must be in Setup). The splits run at lr_scale 1.0, where a random set of
  926,999 rows already costs 0.90 BLEU on news (survives the baseline caveat above). At the rate the
  frozen rule selected (0.15), top-ranked and random sets were not detected as different (gap 0.26,
  t = 2.59, critical 4.30 at floor(df); 3 seeds; smallest detectable gap 0.43 BLEU, 0.34 under the
  exact rule).
- **Compute.** E0.5: 30 runs on one rented 4x RTX 5090 box in one session (2026-10-16, 21:24-00:18
  UTC), Python 3.12.15, torch 2.14.1+cu132.

**Gaps.** (1) BLEU signature to verify (model size is now from the README: 60.5M). (2) The E0.6
sets must be added to the sets table with the same statistics. (3) E0.7 needs its own setup paragraph (new en-de base, Europarl
held-out set); E1.0 needs a recipe paragraph (prompt, LoRA, learning-rate pick, MetricX).

---

### 3.4 Results (about 2.3 pages)

**Purpose.** Report every registered contrast with its verdict, in the order: phenomenon in our
setting -> what the set contains -> within-set split -> inside one source -> residual vs random ->
[en-de] -> [LLM].

**4.1 Top-ranked vs random at two learning rates.**
In hand: at lr_scale 0.15 (pre-registered, E0.3; averaged last-5 checkpoints): top 35.17 +- 0.16,
random 35.43 +- 0.07, bottom 32.23 +- 0.13; gap 0.26 not detected (t = 2.59, df 2.6, critical 4.30
at floor(df); 3 seeds; smallest detectable gap 0.43 BLEU, 0.34 under E0.5's exact rule; derived);
held-out UN 46.85 / 47.34 / 43.12. At lr_scale 1.0 (follow-up, **not pre-registered**, 3 seeds,
final checkpoints): top 33.62 +- 0.11, random 34.41 +- 0.07, bottom 29.49 +- 0.13; gap 0.79
(t = 10.62, df 3.5, critical 3.18 at floor(df)); held-out UN top 45.27 (-1.56), random 47.35
(+0.52), bottom 36.65. The domain-shift reading predicts an in-domain gain for a set that is 77.0% UN (the 926,999 rows
trained on; 78.3% is the 1M before deduplication, 3.3); the in-domain score falls. State both caveats: the two
learning rates also differ in checkpoint averaging, and E0.3's step-matched arms differed by up to
1.2013x in applied target tokens.
Gap: the within-set splits were never run at 0.15. E0.6-LR is proposed, not decided (prereg E0.6 OD
17; WMT2027_PLAN risk 1): the UN triple at lr_scale 0.15, 5 seeds, 15 runs, about 6 GPU-hours
(derived), registered separately. Only the triple lets Q2 be read against random text at that rate.

**4.2 What the top-ranked set contains.**
In hand (PROTOCOL E0.4; 926,999-row sets, sampled statistics on 200,000 rows; which statistics were
sampled is **to check**; top / random / bottom):
type/token ratio 0.007 / 0.016 / 0.034; duplicate source rows 4.14% / 0.97% / 7.44%; repeated
8-grams 5.93% / 4.66% / 3.84%; source-length p90 35 / 47 / 40 tokens; a UN-heading regex fires less
in the top set (0.22% vs 0.63%), so "boilerplate" is the wrong word. At identical word counts (E0.5
table): source word types 38,783 (non-repetitive half) / 27,524 (repetitive half) / 94,897 (random).
Before matching, recurring rows average 19.0 tokens against 28.6 for non-recurring rows.
Gap: the type/token ratios are not length-controlled (PRIOR_ART risk 10); report the identical-word-
count figures in the body and the ratios in the appendix. A descriptive depth sweep (at the depths
listed in WMT2027_PLAN section 3.2 item 5) is not computed (section 7, objection 3).

**4.3 Within the top-ranked set, all sources.**
In hand: E0.4 (3 seeds): non-repetitive 32.77 (0.20), repetitive 30.05 (0.28), gap 2.73, floor 0.28,
t = 13.96, df 3.6, critical 3.18 at floor(df) (passes under either lookup). E0.5 replication on
another machine and build (5 seeds): 33.05 vs 30.01, gap 3.04.
The halves differ in source mix (74.2% vs 90.9% UN) and in pretraining exposure (52.9% vs 42.4%;
PROTOCOL dry-run table, re-derive), so
this result alone does not separate the split from source mix; that motivates 4.4.
Do not use in the body: "the repetitive half is within 0.55 BLEU of the bottom band" (E0.4); the
halves are a quarter of the size at the same steps, so they make about four times as many passes
(README states this caveat).

**4.4 Inside one source (the main result).**
In hand: E0.5 Q1 YES. UN only, 214,489 rows each: non-repetitive 30.58 (0.14), repetitive 27.95
(0.11); gap 2.63, Welch t = 33.3, df 7.5; floor 0.14; smallest callable gap 0.18; seeds do not overlap
(27.79-28.09 vs 30.38-30.76). Pretraining exposure 36.7 / 36.7% as exact pairs (measured, not
matched by design; PROTOCOL dry-run table, re-derive). The halves still differ in source
word types (33,874 vs 25,134) and duplicate sources (0% vs 8.45%): "repetitive" is a bundle (say so
here; 3.5 tests its strands without isolating any single one).

**4.5 The non-repetitive half against random text.**
In hand: E0.5 Q2 outcome B. Non-repetitive 30.58 vs random UN 31.17; gap 0.58, t = -7.0, df 7.9,
p < 0.001; smallest callable 0.19; no seed overlap (30.38-30.76 vs 31.02-31.31). In domain (reported,
not decisive): non-repetitive 44.72 (-2.11), repetitive 44.18 (-2.65), random UN 47.03 (+0.20);
all-source halves 44.60 (-2.23) and 44.30 (-2.53), random 46.80 (-0.03). The halves differ by 0.54
in domain against 2.63 out of domain.
Caveats to state: the random UN set is not QE-matched (mean QE 0.8629 vs 0.8990; vocab memo,
scratch) and a quarter of its rows contain an 8-gram that recurs within the same-size random set
(the statistic computed in that set's own parent, `ft_random`, not in the top-ranked set as 0.2
defines it; R0's "no recurring 8-gram" has the same basis), so 0.58 may understate the
"top-ranked minus repetition" cost; E0.6 adds a recurrence-free random set (R0).

**4.6 English-German [E0.7 slot].**
Design (ende memo recommendation, pending prereg E0.7 OD 1-4; every en-de number here is scratch:
built under Python 3.14 / Unicode 16, and the run of record must be rebuilt on the training box):
Europarl only; top 1,000,000 rows by CometKiwi (23.6% of the 4,238,227-row en-de pool), 994,881
after exact-pair dedup (23.5%; derived; `s2_report`'s `share_of_pool` 0.2359 uses the 1M selected
rows; QE 0.8599-0.9146); three sets of 104,521 rows (non-repetitive, repetitive, random Europarl;
at the 1M band the Europarl repetitive half has 1.03% duplicate sources, against 8.45% for the
en-fr UN one; `s2_report` `duplicate_source_fraction` 0.01031),
same length histogram; 5 seeds, 15 runs; 2,000-row held-out Europarl set removed from base training
and pool; new en-de base trained on the corrected corpus minus held-out rows (every arm then 100%
seen in pretraining). Step count open (10,000 steps = 18.4 passes over 104,521 rows vs about 4,900 =
9.0 passes, en-fr's count).
Slot: "[Q1-ende: gap {g}, t {t}; Q2-ende: gap {g}, t {t}; held-out Europarl {..}]".
How the slot is read: prereg E0.7's opening paragraphs and its Q1-de / Q2-de outcomes. One strength to state: before matching, the en-de repetitive half is longer than the non-repetitive half
over all sources at the 1M band (26.6 vs 20.4 tokens; ende memo, `s2_report`
`halves_before_matching`), the reverse of en-fr over all sources (19.0 vs 28.6). The design is
Europarl only, so the paper needs the Europarl figure: the pre-registration draft derives 29.2 vs
21.5 from the memo's scratch files (**to check**; re-derive on the box before quoting). Only then
may the paper say that a Europarl gap cannot come from short text.

**4.7 A 1.7B LLM [E1.0 slot].**
Design (llm memo): Qwen3-1.7B-Base; LoRA on the three UN sets, 5 seeds (15 runs); LoRA learning
rate picked on the random set only from {1e-4, 2e-4, 4e-4} on FLORES+ dev (3 runs); full fine-tuning
repetitive vs non-repetitive, 3 seeds each at 2e-5 (6 runs); a second model with LoRA, repetitive
vs non-repetitive, 3 seeds each (6 runs): Gemma-3-4b-pt (the llm memo's and the pre-registration
draft's recommendation) or Qwen3-4B-Base, or another use of the 6 runs, not yet decided (prereg E1.0
OD 1). E1.0 covers two model families only if the second model
is Gemma; Qwen3-4B-Base is the same family as Qwen3-1.7B-Base. Primary: WMT24++ en-fr_FR with MetricX-24-hybrid-XL;
secondary FLORES+ devtest, held-out UN, newstest2014 (continuity only), chrF and BLEU; COMET-22
reported and flagged. Pilot gate: reproduce the published 8-shot FLORES+ en-fr score for
Qwen3-1.7B (39.7 spBLEU / 83.8 COMET, Shang et al. Table 5). **To check** (llm memo): Shang et al. do
not say whether "Qwen3-1.7B" is the base or the instruct model, and the memo matched numbers to
models by column order, so the gate may compare the base model against an instruct-model number.
Slot: "[Q1-LLM: {MetricX gap}; Q2-LLM: {gap}; full fine-tuning: {gap}; second model ({name}): {gap}]".
Readings: prereg E1.0, Outcomes (Q1-LLM, Q2-LLM with both directions, F-Q1, G-Q1); Pombal 2025 and
Fujita de Mello 2025 found CometKiwi top-k beating random for LLMs.

---

### 3.5 Analysis (about 1.9 pages)

**Purpose.** Test the strands of the "repetitive" bundle (no E0.6 contrast isolates a single one)
and say what else selection changes. This section is where objections 1 and 2 are answered with
experiments; objection 4 is answered by claiming no remedy (5.5).

**5.1 Beyond vocabulary coverage? (answers objection 1, Saunders & DeNeefe 2024)**
In hand: source word types at identical word counts, UN triple: 25,134 (repetitive) / 33,874
(non-repetitive) / 58,441 (random). E0.5 matched size, length and source; exposure was measured
balanced, not matched by design; vocabulary was not matched.
E0.6 contrast: `ft_un_rep` vs `div_narrow` (prereg: C1-beyond-coverage). `div_narrow` is drawn from
the 486,475 non-recurring UN candidates at `ft_un_rep`'s exact length histogram and narrowed to
`ft_un_rep`'s coverage by the rule fixed in prereg E0.6 OD 2. The measured range at that histogram
runs from 20,284 source word types (in every length bucket, the rows whose rarest word is most
frequent) to 34,070 (a random draw), and 25,134 lies inside it (feasibility memo, Mac scratch;
whether the chosen rule reaches ±2% is not measured; rebuild with a fail-closed script). It holds
size and length fixed and matches coverage on the OD 1 measure within ±2%; what it leaves free is listed in prereg E0.6, "What follows from
the table" (exact duplicates, French-side features, whatever rarest-word selection shifts), and the
paper names all of it beside the result. Its coverage measure is prereg E0.6 OD 1; the ninth set
that would also remove duplicates is OD 5.
Also before pre-registration, CPU only: Saunders & DeNeefe's coverage (share of the base model's
subword vocabulary that occurs in the set, source and target) on all E0.5 and E0.6 sets.
Slot: "[repetitive vs div_narrow: gap {g}, verdict, smallest detectable gap]", worded as the reading
prereg states for that verdict (E0.6, Outcomes, C1-beyond-coverage). If prereg E0.6 OD 18
registers V (`ft_un_div` vs `div_narrow`, no extra runs), its result and registered reading go
beside it; whether and how any vocabulary account may be presented is fixed there, not here.
Either way, note that Saunders & DeNeefe's own mix-in result does not separate coverage from
adding generic data (3.2(c)).

**5.2 Beyond exact duplicates? (answers objection 2, "just deduplicate")**
In hand: the repetitive UN half has 8.45% duplicate sources, while every one of its rows contains a
recurring 8-gram (by construction); duplicates are a minority strand of what the split selects.
E0.6 contrast: `rep_dedup` vs `div_dd`, 198,333 rows each (prereg: C1-beyond-duplicates).
`rep_dedup` is the 227,406 repetitive UN candidates with repeated source sentences removed (207,561
left), length-matched against the non-recurring candidates; 0.00% duplicate sources.
Recurring-8-gram share (`ft_un_rep`: 0.433) and source types: the feasibility memo's 0.379 and
25,296 come from a feasibility draw that was not recorded (no script or index survives); a scratch
rebuild to the pre-registration draft's definition (`ft_topk` file order, first copy kept,
stratified matching with `random.Random(42)`; Mac, Python 3.14.3) gives 198,333 rows, 0.378 and
25,256 types for `rep_dedup` and 33,098 types for `div_dd` (deduplicating the diverse side too drops
252 rows and changes nothing). Quote the rebuilt values, marked scratch, until `e06_build.py`
replaces them. So `div_dd` has about 7,800 more source types than `rep_dedup`: this contrast is not
vocabulary-matched. If prereg E0.6 OD 5(b) adds the ninth set, `div_narrow_dd` vs `rep_dedup`
(prereg: C1-beyond-both) holds exact duplicates and the OD 1 coverage measure fixed together;
what it still leaves free (French-side features and whatever rarest-word selection shifts) is
prereg's C1-beyond-both row.
Placement, agreed across the drafts: near-duplicate removal (Waldendorf used BiCleaner
`--aggressive_dedup`) is WMT 2027 work (WMT2027_PLAN section 3.9, W4 arm 5); E0.6 answers exact
duplicates only (prereg E0.6 OD 4).
Slot: "[rep_dedup vs div_dd: gap {g}, verdict]" (and "[div_narrow_dd vs rep_dedup]" if added),
worded as prereg states (E0.6, Outcomes, C1-beyond-duplicates; C1-beyond-both).

**5.3 What else does selection change? (the vocabulary-capped random set N vs the rest S)**
In hand (vocab memo, scratch): R0 (random UN, no recurring 8-gram, `ft_un_div`'s length histogram)
62,000 source types; A_narrow (not a subset of R0: drawn from the same random-UN rep = 0 pool, 450,709
rows, restricted to rows whose source words are all among the pool's 42,000 most frequent and French
words among its 58,000 most frequent, leaving 402,010; each `ft_un_div` row is matched within its
length bucket by the pretraining rank of its rarest word) 33,562 types against 33,874 for
`ft_un_div`; QE 0.861 / 0.866 / 0.899; word-for-word literalness 0.693 / 0.698 / 0.783. A_narrow
shares 136,916 rows with R0 (scratch: intersection of `vocab/R0.npy` and `vocab/A_narrow_42_58.npy`;
vocab memo risk 7 prints "136,9k"; `e06_build.py`'s manifest replaces it): the contrasts stay
valid but measure partial swaps.
E0.6 contrasts (prereg contrasts table): precondition P = R0 - `ft_un_div` (= N + S); N = R0 -
A_narrow; S = A_narrow - `ft_un_div`; and the bridges Q1′, Q2′. What each leaves free (A_narrow's
11.7% top-band rows against R0's 9.1%, its unmeasured French recurrence, which words it lacks; S's
lumping of QE level, literal or formulaic French and genre) is in prereg E0.6, "What follows from
the table". At E0.5's UN seed sds (0.11-0.14) the rule called gaps down to 0.18-0.19 BLEU with
5 seeds; at a seed sd of about 0.13, about 0.26 BLEU has 80% power (vocab memo; prereg E0.6, "Why
five seeds").
Target-side descriptives for the discussion (vocab memo, scratch; rebuild before citing): French
types at 6.5M tokens 43,021 (non-repetitive) / 32,684 (repetitive) / 69,320 (random UN) / 72,574
(R0); rows containing a recurring French 8-gram 23.0% / 79.7% / 29.0% / 15.0%; word-for-word
literalness 0.783 / 0.810 / 0.714 / 0.693. So the non-repetitive half has no recurrence on the
source side, yet its French recurs more than R0's (23.0% vs 15.0%) and is more literal than either
random set. That is an alternative to narrowness. Literalness varies only in S (A_narrow 0.698 vs
R0 0.693). A_narrow's French recurrence is not measured (scratch `vocab/a4.py` covers only the two
halves, random UN and R0; **to check**); with 43,703 French types against R0's 72,574 its French may
recur more than R0's, so N may carry part of the formulaic-French strand.
Slot: "[P {g}; N {g}; S {g}]", each with its verdict and smallest detectable gap, worded as prereg
states them (E0.6, Outcomes: precondition P, N, S, Joint readings, Bridges). Those readings apply
only if P is detected and Q2′ replicates; otherwise N and S are reported by their verdicts alone.
The paper never apportions P between N and S (prereg E0.6: the N and S outcomes, "never as a share",
and the paragraph "No sentence of the form ...") and never writes "not narrowness" (prereg S+).
Design C (literal vs free split): WMT 2027 work, run only if S is detected and E0.6 gives S a reading: P holds and Q2′ replicates
(prereg E0.6, *Precondition P*, S+ and *Bridges*), registered separately (WMT2027_PLAN section 3.3, follow-up arms); not in this paper.

**5.4 In domain vs out of domain.**
In hand: 4.5's numbers. The reading "the whole top-ranked set is lexically narrow (costing in and out
of domain) and its repetitive part adds an out-of-domain cost" fits every number and is untested
(README). E0.6 secondary readout: all sets on held-out UN, deciding nothing; the one reading stated
in advance, and why the vocab memo's "genre or style, not vocabulary" is not used, are in prereg
E0.6, "Reported and deciding nothing" (held-out UN). Facts to quote with it: A_narrow 98.55% of
held-out UN tokens seen at least 5 times, R0 98.85%, `ft_un_div` 97.75% (vocab memo, scratch).
Slot: "[held-out UN for R0, A_narrow, div_narrow, rep_dedup, div_dd]".

**5.5 Accounting and replication.**
In hand: token bounds (3.3); E0.4 replicated on a second machine and a rebuilt split (3.04 vs 2.73);
scoring reproduces and is 99.10% bit-identical on a 5,000,000-row repeat. E0.6 re-runs the three
E0.5 UN sets as a bridge; no E0.5 number is pooled with E0.6.

**Answer to objection 4 (NRC re-ranking as the baseline for any remedy).** The paper proposes no
remedy (ledger: not claimable). One sentence of evidence that is honest about its limits:
`ft_un_div` is a length-matched sample (214,489 of 486,475) of the top-ranked UN rows that contain no
recurring 8-gram, and it still trails random UN text by 0.58; so within-set redundancy removal alone did not reach random-level quality.
That is not NRC's procedure (bigram novelty with refill from lower ranks), so it is not the baseline
comparison. Placement, agreed across the drafts: the remedy and its NRC baseline are WMT 2027 work
(WMT2027_PLAN section 3.9, W4 arms 3-4); E0.6 has no NRC arm (prereg E0.6 OD 13).

---

### 3.6 Limitations (about 0.8 page)

Each item is a sentence in the paper. Items marked * shrink if the named experiment reports.

1. One QE metric (CometKiwi-22). No claim about QE metrics in general, even after E0.7.
2. One curated pool per pair; the trained top set is 77.0% UN in en-fr (78.3% of the 1M before
   deduplication). *E0.7 adds a Europarl-dominated pool.
3. Extreme top only: the top 1,000,000 rows, 6.0% of the pool in en-fr (5.6% after exact-pair
   dedup) [and 23.6% in en-de (23.5% after dedup; scratch)]. Peter et al. 2023 found the
   QE-best 50% beats random from scratch; our results do not contradict that.
4. Fine-tuning an already trained model; the English-French splits run at a learning rate
   where random data also costs news BLEU; at the rule-selected 0.15 no top-vs-random difference was detected (3 seeds; smallest detectable
   gap 0.43 BLEU), and the splits were not run there. *E0.6-LR (proposed, not decided; prereg
   E0.6 OD 17) would run the UN triple there.
5. One base model per pair. *E1.0 adds one LLM family (Qwen3), two only if its second model is
   Gemma-3-4b-pt (prereg E1.0 OD 1).
6. NMT arm scored with BLEU on newstest2014 and one in-domain set only. Re-scoring the 30 kept E0.5
   checkpoints with chrF, a neural metric and further test sets is possible after a re-decode, since
   their outputs were not kept (section 9, N1).
7. "Repetitive" is a bundle (recurring n-grams, duplicates, thin vocabulary, more formulaic French).
   *E0.6 tests the repetitive half (with its duplicates) against a set coverage-matched on prereg
   E0.6 OD 1's measure, the duplicate-free repetitive half against a non-repetitive draw, and a
   vocabulary-capped random set against the rest of selection (N and S measure partial swaps); no
   single strand is isolated; genre and literalness stay lumped in S. If prereg E0.6 OD 5(b) adds
   the ninth set, one contrast (`rep_dedup` vs `div_narrow_dd`) holds exact duplicates and the OD 1
   coverage measure fixed together; French-side features and the side effects of rarest-word
   selection stay free even there (prereg, C1-beyond-both).
8. The recurrence statistic as run differs from its registered wording for 59 rows (44 change
   sides); the split follows the interpreter's Unicode tables, so it is reproducible per interpreter,
   not across versions (E0.4 note).
9. A median split is a two-point contrast; no dose-response, no "predicts".
10. The lr-1.0 top-vs-random comparison (0.79 BLEU) was a follow-up, not pre-registered.
11. For E1.0: newstest2014 and UN text are probably in the LLM's pretraining; only between-condition
    contrasts are claimed; WMT24++ references date from February 2025 (llm memo).

---

### 3.7 Conclusion (about 0.3 page)

**Purpose.** Restate only ledger rows marked "supported" plus whichever slots filled. One sentence
on what practitioners can take away, scoped: when fine-tuning on the very top of a QE ranking,
inspect recurrence and vocabulary against a random draw of the same pool. One sentence on WMT 2027:
remedy against NRC re-ranking, selection-depth sweep, second QE metric. No new claims.

---

### 3.8 Appendices (not in the 10 pages; limits **to check**)

- **A. Deviations from pre-registration** (title exactly as PROTOCOL 6.6 requires; the section appears
  whether or not it is empty; it also carries the registration record). E0.3: Europarl dropped from
  the gate, a recorded deviation forced by D4 (every candidate pair was in the pretraining corpus);
  amendment X1, made before any result (criterion 2 must exceed that set's seed sd, mirroring
  criterion 1's noise floor). E0.4: length-matching rule amended before any run (first rule did not
  converge); final checkpoint instead of D8's averaged checkpoint (to match the lr-1.0 control); first
  launch inherited a loss-spike guard of 1.3 against D9, stopped after 20 minutes, the one completed
  run and the partial second deleted unused and all six restarted (the guard had fired zero times);
  statistic wording vs code (59 rows, 44 sides); E0.4 input differs from E0.3's top set by one row of
  926,999. E0.5: registered at commit `ecdd6f8`, then
  amended before any run at `0fb52a6` after a second pre-flight (pretraining exposure added as exact
  pairs, the vocabulary wording corrected to 41% / 58%, the rule that no claim is drawn from the
  all-source vs UN gap size, Q2's glosses conditional on Q1 = YES; see that commit's message); token
  imbalance above the 2% flag, known and bounded in advance. Registration dates and commits (E0.3
  decisions frozen 2026-09-18, tag `e03-decisions-v1`; E0.4 2026-09-20; E0.5 tag `e05-prereg-v1` at
  commit `0fb52a6`); how to cite them anonymously **to check**.
- **B. Per-seed tables.** E0.3, lr-1.0 probe, E0.4, E0.5 (and E0.6, E0.7, E1.0).
- **C. Set construction.** Recurrence statistic and BLAKE2b keys, stratified matching, exact-histogram
  random draws, slack, overlap counts, set hashes.
- **D. Token accounting.** Applied target tokens per run; the bound computation.
- **E. Learning-rate selection.** The stage-1 traces (README table) and the full-pool sensitivity
  (no rung passes).
- **F. Examples.** Five repetitive and five non-repetitive UN rows (to extract from the set files).
- **G. Corpus rebuild.** The carriage-return issue in one paragraph; legacy mode byte-identical;
  exact-match provenance 99.40% (a lower bound).
- **H. E0.6 / E0.7 / E1.0 construction and recipes.**
- **I. Reproducibility.** Software versions, scorer revisions, hashes, release plan.

---

## 4. Figures and tables

| id | shows | made from | status |
| --- | --- | --- | --- |
| **T1** | The fine-tuning sets: rows, UN share, recurring-8-gram share, source word types at identical word counts, duplicate sources, share in pretraining, mean QE | `~/mt_local/phase0_final/e05_results/data/phase0_e05/manifest.json` (rows, sources, rep, types, duplicates); pretraining share re-derived from the run-of-record set files in `e05_results/data/phase0_e05` (the PROTOCOL E0.5 table is the Mac dry run); mean QE re-derived from `~/mt_local/phase0_final/v2_scored.tsv` (vocab memo values are scratch); E0.6 rows from the E0.6 build manifest | E0.5 rows ready; E0.6 rows pending |
| **T2** | Main result: the six E0.5 sets on newstest2014 and held-out UN, mean (sd), change from base; the three registered contrasts with gap, t, df, floor, smallest callable gap; caption: "vs base" is biased toward "degrades" by ~0.2-0.4 BLEU (averaged baseline vs single final checkpoints), contrasts unaffected | `~/mt_local/phase0_final/e05_results/results/e05_probe.tsv` (per seed) and `e05_report.txt` (verdicts) | ready |
| **T3** | Top-ranked vs random vs lowest-ranked at lr_scale 0.15 and 1.0, both test sets | `~/mt_local/phase0_results/phase0_bleu.tsv` (+ `.meta.json`, `lr_selection.json`); `~/mt_local/phase0_results/lr1_probe/lr1_probe.tsv` (+ meta) | ready; footnote the checkpoint-averaging difference |
| **F1** | Per-seed newstest2014 BLEU for the six E0.5 sets, baseline as a line (shows the non-overlap) | `e05_probe.tsv` | ready |
| **F2** | Out-of-domain change vs in-domain change per set (newstest2014 delta against held-out UN delta); caption carries the same baseline caveat as T2 | `e05_probe.tsv`; E0.6 report when available | E0.5 points ready; candidate to move to appendix |
| **T4** | E0.6 contrasts, with verdict and smallest detectable gap: precondition P (R0 vs `ft_un_div`), N, S, repetitive half vs coverage-matched (`ft_un_rep` vs `div_narrow`), duplicate-free repetitive vs non-repetitive (`rep_dedup` vs `div_dd`), [V, `ft_un_div` vs `div_narrow`, only if prereg E0.6 OD 18 registers it], [`div_narrow_dd` vs `rep_dedup`, only if OD 5(b) adds it], and the bridges Q1′ and Q2′ to E0.5 | E0.6 report file (name fixed at pre-registration, **to check**) | pending E0.6 |
| **T5** | en-de Europarl triple, both test sets | E0.7 report and build manifest; scratch descriptives in `phase0/next/ende/s2_report_*.json`, `s3_report.json` until rebuilt | pending E0.7 |
| **T6** | LLM arm: MetricX, chrF, BLEU on WMT24++, FLORES+, held-out UN; LoRA, full fine-tuning, second model (prereg E1.0 OD 1) | E1.0 outputs (names fixed at pre-registration) | pending E1.0 |
| **F3** (optional, appendix) | How the top of the ranking changes with depth: recurring share, types, duplicate rate, UN share at the depths listed in WMT2027_PLAN section 3.2 item 5 | `v2_scored.tsv` plus the `e04_split.py` statistic (CPU, not computed); en-de bands from `s2_report_*.json` (scratch) | not computed |
| A-tables | E0.4 per seed; applied tokens per run; stage-1 traces; deviations | E0.4: `e04_probe.tsv` on the author's box (not in `~/mt_local`; location **to check**); tokens: `e05_probe.meta.json` (`budget`); traces: README table; per-run logs `e05_results/logs/phase0_e05/*.log` | E0.4 file to locate |

---

## 5. Claims ledger

Status: supported / pending E0.6 / pending E0.7 / pending E1.0 / not claimable.

| # | claim (as the paper would word it) | status | evidence and caveats |
| --- | --- | --- | --- |
| 1 | The highest-scoring slice can train no better, or worse, than a same-size random draw, a lower-scoring band or the full data (credited to prior work, not claimed) | supported (by citation) | Waldendorf 2025 and Chimoto 2024 (same-size random); Tan 2024 (lower QE band); Finkelstein 2024 (full data, no random control) |
| 2 | In our setting, at lr_scale 1.0, fine-tuning on the top-ranked set costs 0.79 BLEU more on news than a random set of the same size | supported | lr-1.0 probe: 33.62 vs 34.41, t = 10.62; 3 seeds; **not pre-registered** |
| 3 | At the rule-selected rate (0.15), no top-vs-random difference was detected | supported | E0.3: gap 0.26, t = 2.59, crit 4.30 (floor(df)); 3 seeds; smallest detectable gap 0.43 BLEU (0.34 under the exact rule; derived after the fact, not registered for E0.3); pre-registered |
| 4 | The domain-shift explanation does not fit: top-ranked text lowers in-domain BLEU while random data does not | supported, descriptive only (not a registered contrast; held-out UN decided nothing in E0.5; lr-1.0 probe not pre-registered) | lowers it: lr-1.0 top-ranked set (77.0% UN) -1.56, E0.5 UN halves -2.11 / -2.65; random data does not: +0.52 at lr-1.0, +0.20 (random UN) and -0.03 (`ft_random_small`) in E0.5; all "vs base", see the baseline caveat (3.3) |
| 5 | The top-ranked set is lexically thinner and more repetitive than a same-size random draw | supported (one pool) | E0.4 descriptives against `ft_random` of the same size, not length-matched (p90 length 35 vs 47); type/token ratio not length-controlled. The identical-word-count figure 38,783 vs 94,897 is the non-repetitive HALF vs `ft_random_small`, not the whole set; a whole-set vs length-matched random comparison is not computed for en-fr (CPU) |
| 6 | The top-ranked set's scores form a narrow plateau (0.8959-0.9145) | supported | E0.3 |
| 7 | Within the top-ranked set, the repetitive half costs more on news than the non-repetitive half | supported | E0.4 2.73 (t = 13.96); E0.5 replication 3.04; source mix differs (74.2 vs 90.9% UN) |
| 8 | Inside one source, at matched size and length, with QE and the exposure share measured nearly equal (exposure frequency is not: 1.056 vs 0.951 per row), the repetitive half costs 2.63 BLEU more | supported | E0.5 Q1, t = 33.3, pre-registered; QE 0.8990 vs 0.8993 (scratch), exposure 36.7 vs 36.7% as exact pairs |
| 9 | The repetitive half's cost is not reducible to source mix | supported | claim 8; registered wording "Repetitiveness is not reducible to source mix" (e05_report.py, README), a verdict about the split; the repetitive half is a bundle (fewer word types, 8.45% duplicate sources) whose strands E0.5 did not separate; the all-source gap may still include source-mix and exposure effects (collinear there, PROTOCOL) |
| 10 | The repetitive half (recurrence with its duplicates) costs quality beyond vocabulary coverage (prereg E0.6 OD 1's measure) | pending E0.6 | `ft_un_rep` vs `div_narrow`, which also differ in exact duplicates (8.45% vs none expected; confirm in the build manifest), so it tests the repetitive half (recurrence with its duplicates) against a coverage-matched set, not recurrence alone; target-side features and the side effects of choosing rows by their rarest word's frequency also differ (5.1); row 11 removes duplicates but is not coverage-matched; if prereg E0.6 OD 5(b) adds the ninth set, `rep_dedup` vs `div_narrow_dd` holds exact duplicates and the OD 1 coverage measure fixed together; coverage measure: OD 1; reading: prereg C1-beyond-coverage |
| 11 | With exact duplicates removed, the repetitive half still costs quality | pending E0.6 | `rep_dedup` vs `div_dd`; `div_dd` is a diverse draw of the same size, not coverage-matched (scratch rebuild, to be replaced by `e06_build.py`: `rep_dedup` 25,256 types, `div_dd` 33,098; the feasibility memo's 25,296 is from an unrecorded draw), so recurrence and vocabulary still move together unless prereg E0.6 OD 5(b) adds the ninth set (`rep_dedup` vs `div_narrow_dd`, coverage-matched on the OD 1 measure); near-duplicates not covered (W4); reading: prereg C1-beyond-duplicates |
| 12 | The non-repetitive half trails size- and length-matched random text from the same source (not QE-matched) by 0.58 BLEU | supported | E0.5 Q2, t = -7.0; random arm not QE-matched; 25% of its rows contain an 8-gram that recurs within its own same-size random parent (`ft_random`), not within the top-ranked set |
| 13 | N (the vocabulary-capped random set, A_narrow, against R0) and S (the rest of selection: A_narrow vs `ft_un_div`), each detected or not detected with its smallest detectable gap | pending E0.6 | N and S contrasts; for what N may be read as (narrower vocabulary plus whatever else the caps change), see prereg E0.6, N; they measure partial swaps (A_narrow shares 136,916 rows with R0, scratch); S may absorb part of a vocabulary effect (vocab memo risk 3); the gap is not apportioned between them (prereg E0.6, "never as a share" and "No sentence of the form ..."); read only as prereg states: only if P is detected and Q2′ replicates, otherwise by verdict alone (5.3) |
| 14 | The repetitive-vs-non-repetitive gap is much smaller in domain than out of domain | supported, descriptive only | E0.4: 0.11 in domain (not detected; smallest detectable gap about 0.38 at floor(df), 0.35 exact, from the README's rounded sds; derived, recompute from `e04_probe.tsv`) vs 2.73; E0.5: 0.54 vs 2.63; in-domain contrasts were not decision contrasts |
| 15 | The whole top-ranked set is narrow; its repetitive part adds an out-of-domain cost | not claimable | an untested reading (README: "fits; it has not been tested"); prereg E0.6 gives it no route to support (held-out UN is reported and decides nothing; a detected N "does not show it is the top band's mechanism"); at most mentioned as untested in 5.4 |
| 16 | The token differences between arms do not explain the gaps | supported | bounds 0.043 / 0.034 BLEU vs floor 0.14 |
| 17 | The result replicates on English-German | pending E0.7 | to be registered as a weaker test before the base is trained; the reading of a null is prereg E0.7, Q1-de |
| 18 | The en-de top set also over-selects recurring rows relative to random | pending E0.7 | scratch: 9.28% vs 4.26% rows with recurrence (6% band), types 44% of random; rebuild as run of record |
| 19 | The result holds for a 1.7B LLM (LoRA; full fine-tuning; second model, a second family only if Gemma, prereg E1.0 OD 1) | pending E1.0 | Q2 may reverse; prereg E1.0 Q2-LLM states both directions (llm memo, open question 1) |
| 20 | QE metrics in general select repetitive text | not claimable | one metric |
| 21 | First to observe the phenomenon or the redundancy | not claimable | PRIOR_ART verdict |
| 22 | Recurrence explains x% of the top-vs-random gap | not claimable | this paper's own rule (WMT2027_PLAN section 1.4 item 8); E0.5 registered only the narrower ban on reading the all-source gap against the UN one ("Repetitiveness explains x% of E0.4"), because sizes, passes and exposure differ between triples (0.3) |
| 23 | A recurrence statistic predicts harm | not claimable | two-point contrast |
| 24 | A diversity or redundancy control fixes QE selection | not claimable | no intervention; NRC re-ranking is the required baseline |
| 25 | The finding holds for training from scratch or for keeping 50% | not claimable | fine-tuning only; Peter 2023 |
| 26 | The finding holds at other learning rates | not claimable | splits run only at lr_scale 1.0; E0.6-LR (UN triple at 0.15) proposed, not decided (prereg E0.6 OD 17) |
| 27 | The finding holds under chrF / neural metrics / other test sets (NMT arm) | not claimable now | not run; possible by re-decoding and re-scoring kept checkpoints (section 9, N1) |

---

## 6. Page budget (10 pages)

WMT2027_PLAN section 2.1 follows this structure and these budgets.

| section | pages | floats |
| --- | --- | --- |
| Title, abstract | 0.3 | |
| 1 Introduction | 1.0 | |
| 2 Related Work | 1.2 | |
| 3 Setup | 1.4 | T1 |
| 4 Results | 2.3 | T2, T3, F1; T5 and T6 compact if E0.7 / E1.0 report |
| 5 Analysis | 1.9 | T4, F2 |
| 6 Limitations | 0.8 | |
| 7 Conclusion | 0.3 | |
| Slack | 0.8 | absorbs T5/T6 |
| **Total** | **10.0** | |

If over, cut in this order: F2 to the appendix; 4.3 to two sentences (E0.4 details to the appendix);
Related Work (d) and (e) to one sentence each; 4.2's sample statistics to the appendix. Never cut
the learning-rate regime from Setup or the scope sentence from the Introduction. If references count
toward the 10 pages (**to check**), take the space from slack and Related Work (e).

---

## 7. The four reviewer objections, and where each is answered

| # | objection | answered where | with what | still open |
| --- | --- | --- | --- | --- |
| 1 | Saunders & DeNeefe 2024: forgetting tracks vocabulary coverage, so "repetitive" may stand in for low coverage | 5.1 (experiment), 2(c) (positioning), 6 item 7 | E0.6 `ft_un_rep` vs `div_narrow` at matched size and length, coverage-matched on prereg E0.6 OD 1's measure; their coverage statistic on every set (CPU) | which coverage measure to match (prereg E0.6 OD 1); how each outcome of `ft_un_rep` vs `div_narrow`, and of V if OD 18 registers it, is read: prereg E0.6, Outcomes and OD 18 (5.1) |
| 2 | "Just deduplicate" (Koehn 2018; Waldendorf 2025) | 5.2 (experiment), 2(a)/(b) | E0.6 `rep_dedup` vs `div_dd` (198,333 rows each; `rep_dedup` has 0.00% duplicate sources and keeps a recurring-8-gram share of 0.378 in a scratch rebuild, 0.379 in the feasibility memo's unrecorded draw; 5.2); the repetitive half's duplicates are only 8.45% of its rows | near-duplicate removal is WMT 2027 work (W4 arm 5; prereg E0.6 OD 4); vocabulary not matched unless prereg E0.6 OD 5(b) adds the ninth set |
| 3 | Peter et al. 2023: QE filtering beats random at 50% kept, trained to convergence | 1 (scope sentence), 3.3 (band share, regime), 6 items 3-4, 2(a) | scoping; E0.3 at 0.15 reported openly; [E0.7's top 1,000,000 rows are 23.6% of its pool, 23.5% after exact-pair dedup (scratch): a wider band on another pair, differing in pool, source (Europarl), exposure regime (100% seen) and passes, so not a second depth on one ranking] | a depth sweep: descriptive version is CPU-only and not computed (F3); a trained sweep is WMT 2027 work |
| 4 | NRC re-ranking (Lo 2018; Bernier-Colborne & Lo 2019) is the baseline for any remedy | 5.5 last paragraph, 7, ledger 24 | no remedy is claimed; `ft_un_div` (a length-matched sample of the top-ranked UN rows that contain no recurring 8-gram) still trails random by 0.58 | none for this paper: an actual bicov arm is WMT 2027 work (WMT2027_PLAN 3.9, W4 arms 3-4; prereg E0.6 OD 13) |

**Further objections the paper must also meet** (not in the four, all from the sources):

- *"Your effect needs an aggressive learning rate."* 3.3 and 6 item 4; E0.6-LR proposed, not decided (prereg E0.6 OD 17).
- *"BLEU on newstest2014 only."* 6 item 6; E1.0 uses MetricX and WMT24++; NMT re-scoring after a re-decode (section 9, N1); E0.6's secondary metrics (prereg E0.6 OD 11).
- *"It is the pool, not the metric"* (the trained top set is 77.0% UN). 4.4 (inside-UN design) and E0.7.
- *"Pool noise"* (Chimoto). The pool is curated and length-matched; language-ID and length-ratio
  rates per half are not reported (CPU check, section 9, N1).
- *"Metric training coverage"* (Waldendorf). One sentence in 2(a); CometKiwi-22's DA coverage for
  en-fr and en-de **to check**.
- *"Fine-tuning gaps are transient"* (Waldendorf learning curves). Dev-BLEU traces exist per run:
  the E0.5 logs carry "Valid BLEU" every 1,000 steps, ten points per run (checked 2026-10-16), and
  prereg E0.6 reports each contrast along that curve, deciding nothing. Show the gap over training
  in the appendix.
- *"'Repetitive' is templated numerals or document codes"* (Lo 2018 note, reviewer risk 5). Recompute
  rep with numbers, dates and UN document codes masked, and report the numeric-token share per half
  (CPU).
- *"Your random arm was drawn once"* (Waldendorf note, risk 5). Each random set was drawn once (seed
  42) and trained with five seeds, while Waldendorf redraws random data per seed. State it; a
  per-seed redraw would need new runs.
- *"It depends on the step count"* (Finkelstein note, risk 4). All arms ran 10,000 steps: about 2.1
  passes for the 926,999-row sets (E0.3 and the lr-1.0 probe; derived, 192 x 10,000 / 926,999, D9:
  every arm stops at 115,000 micro-steps from 105,000), 7.9 for the 242,022-row sets and 9.0 for the
  214,489-row sets; no other step count was run. Concede in Limitations, or point to the dev-BLEU
  traces.
- *"en-de and en-fr scores are one run"* (for E0.7). The en-de QE scores come from a different
  python/torch stack than the en-fr ones (README, "QE scoring inventory"); never quote the two as if
  from one run.

---

## 8. What carries over from WMT2027_PLAN.md

The plan's thesis ("top-k QE filtering is covert domain selection", predicted by a divergence
statistic) was refuted by the lr-1.0 follow-up to E0.3 and by E0.5 (not by the E0.3 gate verdict
itself): the top-ranked set lowers in-domain BLEU.

**Keep.**
- Anti-circularity: select with CometKiwi-22, evaluate with a different family (MetricX-24) plus
  chrF and BLEU; COMET-22 flagged (now E1.0's design).
- Multi-seed reporting with the smallest detectable gap beside every non-detection (E4.1, narrowed).
- Current test sets (WMT24++) next to newstest2014.
- Applied target tokens as the accounting unit (E0.2), reported per run.
- Pre-registration before runs, a "Deviations from pre-registration" section, every registered cell
  reported (PROTOCOL 6.6-6.7).
- Peer-review path only, no arXiv; release manifests, logs and hashes.
- Related Work engages Peter 2023, Finkelstein 2024 and the WMT18 task by name.
- Presentation lessons from the WMT 2026 rejection: hyperparameter appendix, a real Related Work
  section, readable figures; the acl.sty/lmodern font clash applies only if EAMT uses that style
  (**to check**).

**Drop.** The covert-domain-selection thesis; the Moore-Lewis divergence statistic and dose-response
fit; the R/N/Q/D/C orthogonalisation grid; Base vs Big and the run-to-overshoot convergence protocol
(PROTOCOL sections 1-5, 6.1-6.5 and 7, the Big-vs-Base protocol; 6.6-6.7 are kept, above); the
corpus-regime axis; the from-scratch 14-condition grid.

**Fix (citation error).** The plan cites "Zhou & Savova (arXiv 2503.07329)" for seed variance and a
"ten-seed LoRA design". The arXiv page lists Bui, Savova and Wang (IJCNLP-AACL 2025), and the
abstract does not mention LoRA. If the paper cites it at all, cite Bui, Savova & Wang for what the
full text says (**to check**; only the abstract has been read) and drop the LoRA attribution.

---

## 9. Where each open choice is decided, and order of work

**Order, schedule, runs and cost:** E0.6 -> E0.7 -> E1.0, on rented cards that the author rents
when told; dates, run counts, GPU-hours and what must happen first are WMT2027_PLAN sections 3.1,
4 and 7, not restated here.

**Decisions that shape a slot.** Every design choice is an OPEN DECISION in prereg, with its options
and recommendation, answered by the dates in WMT2027_PLAN section 7. The slots depend on these:

| slot in this outline | prereg decision |
| --- | --- |
| 3.2(c), 5.1, objection 1: coverage measure; whether V (`ft_un_div` vs `div_narrow`) is registered | E0.6 OD 1, OD 18 |
| 5.1, 5.2, Limitations 7, ledger 10-11, T4: the ninth set `div_narrow_dd` (40 or 45 runs) | E0.6 OD 5 |
| 5.2, objection 2: what counts as a duplicate; near-duplicates in W4 | E0.6 OD 4 |
| 5.5, objection 4: no NRC arm in E0.6; W4 | E0.6 OD 13 |
| 5.3: whether R0 or `ft_un_random` is the Q2 reference | E0.6 OD 14 |
| 4.1, Limitations 4, ledger 26: E0.6-LR (UN triple at lr_scale 0.15) | E0.6 OD 17 |
| objection "BLEU only": E0.6's secondary metrics and second test set | E0.6 OD 11 |
| 4.6, T5: en-de band, steps, base, random arm | E0.7 OD 1-4 |
| 4.7, T6, Limitations 5, ledger 19: the second model's 6 runs; schedule; paragraph scoring | E1.0 OD 1, OD 4, OD 5 |

**Two items that are this outline's own** (not experiment design):

- **N1. NMT-arm additions without new training.** chrF and a neural metric on the E0.5 outputs,
  FLORES+ devtest decoding of the kept checkpoints (if prereg E0.6 OD 11 picks it; WMT24++ is
  paragraph-level and unsuited to this arm, OD 11), language-ID and length-ratio rates per half,
  and the F3 depth sweep. E0.5's decoded outputs were not kept (only the BLEU cache), so the metric
  items need a re-decode of the 30 kept `final.pt` first; when and at what cost: WMT2027_PLAN
  section 3.2 (items 3, 5, 6) and section 4 (R1, R3).
- **N2. Correct internal wording** that says E0.5 matched the random arm in QE (only the halves are
  equal, and that was measured, not built in; vocab memo question 1; WMT2027_PLAN section 6.2).

**Reading.** What to read when, and before which sentence of this paper, is WMT2027_PLAN section
6.2 (the ten papers in the reading week, with their first checks; the extra reading and its
deadlines). In the reading week, also check the "check before citing" column of 3.2 and confirm
every number this outline quotes from each paper.

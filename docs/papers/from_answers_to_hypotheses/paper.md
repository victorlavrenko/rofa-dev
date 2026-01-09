# From Answers to Hypotheses:

## Parallel Clinical Reasoning as a Decision Paradigm for Medical AI

**Victor Lavrenko**
PeaceTech VC
[victor@peacetech.vc](mailto:victor@peacetech.vc)

**January 7, 2026**

---

## Abstract

Recent advances in large language models (LLMs) have demonstrated expert-level performance on medical question answering and diagnostic benchmarks (Singhal et al., 2025; McDuff et al., 2025). Despite this progress, clinicians and regulators remain concerned that such systems often appear *confidently wrong*, limiting their use in high-stakes medical settings. In this work, we argue that this issue arises primarily from how model outputs are aggregated and presented, rather than from a lack of medical knowledge. Through a simple experiment using multiple independent reasoning traces, we show that LLMs frequently exhibit latent uncertainty that is hidden when only a single answer is returned. We propose reframing medical AI from answer selection to reasoning in a space of clinical hypotheses, analogous to differential diagnosis in medicine. Our analysis reveals that majority voting over reasoning paths provides little benefit, while alternative hypotheses often contain the correct answer. We relate these findings to recent work on multi-agent medical reasoning, uncertainty estimation, and conformal prediction, and outline future directions toward interactive, hypothesis-aware medical AI systems.

---

## Reproducibility resources

- **Reproduction notebook**: `notebooks/20_paper_reproduce.ipynb`
- **Metric implementations**:
  - Paper tables: `rofa/papers/from_answers_to_hypotheses/analysis.py`
  - Core consensus utilities: `rofa/core/metrics.py`
- **Question selection protocol**: v1 (subject-balanced MedMCQA selection with filters and hashes
  defined in `rofa/core/question_set.py` and stored in `question_set.json`).

---

## 1. Introduction: the illusion of “confidently wrong” AI

Large language models have rapidly achieved impressive results on medical examinations and diagnostic benchmarks, in some cases matching or exceeding average physician performance on structured tasks (Singhal et al., 2025; Gu et al., 2025). At the same time, randomized clinical trials suggest that simply giving physicians access to an LLM does not reliably improve diagnostic accuracy (Goh et al., 2024). This tension has reinforced a widespread belief that LLMs are unsafe for medicine because they “hallucinate” answers with unjustified confidence.

However, recent studies on uncertainty estimation and abstention indicate that model confidence is poorly calibrated to correctness (Gao et al., 2025; Wen et al., 2024). These findings suggest that the problem may not be that models lack uncertainty, but that uncertainty is not appropriately exposed or used.

In this paper, we investigate a complementary explanation: **LLMs often consider multiple plausible hypotheses internally, but standard decoding collapses this hypothesis space into a single answer**. As a result, valuable information about uncertainty and alternative diagnoses is lost.

---

## 2. Clinical reasoning as hypothesis management

### 2.1 Differential diagnosis in human medicine

Clinical reasoning is fundamentally hypothesis-driven. Physicians are trained to maintain a *differential diagnosis*—a ranked set of plausible explanations—rather than committing prematurely to a single conclusion. Decisions are refined through counterfactual reasoning (“if symptom X were absent, diagnosis Y would be less likely”), additional tests, and discussion with colleagues.

Empirical studies of diagnostic error show that premature closure—fixating on one hypothesis too early—is a major source of clinical mistakes. Importantly, this occurs not because physicians are unaware of alternatives, but because maintaining and revisiting multiple hypotheses is cognitively demanding, especially under time pressure and emotional stress.

### 2.2 Why AI can help—practically, not philosophically

AI systems are not immune to bias or error, but they do not suffer from fatigue, emotional anchoring, or working-memory limitations. While humans *can* reason about multiple hypotheses, doing so consistently in real clinical environments is difficult. LLMs, by contrast, can maintain several hypotheses in parallel and generate independent reasoning traces without additional cognitive cost.

Recent work on multi-agent medical systems explicitly exploits this idea by simulating multidisciplinary team discussions (Chen et al., 2025; Liu et al., 2025; Zhou et al., 2025). Our work explores a simpler but closely related approach: using multiple independent reasoning paths of a single model as a proxy for a hypothesis space.

---

## 3. Experimental setup: reasoning paths as hypotheses

We treat each independently sampled reasoning trace as a distinct clinical hypothesis. For each multiple-choice medical question, we generate (N = 10) independent reasoning paths.

Let

* (H_i) denote the (i)-th hypothesis,
* (a_i \in {A,B,C,D}) the selected answer.

We define the empirical stability of the leading hypothesis as

[
\text{max_frac} = \frac{\max_a |{ i : a_i = a }|}{N}.
]

This quantity is analogous to consensus strength in a clinical team or multi-agent system.

---

## 4. Hypothesis H1: majority vote improves accuracy

Self-consistency decoding has shown that majority voting over multiple chains of thought can substantially improve performance on reasoning tasks such as math and commonsense QA (Wang et al., 2022). A natural hypothesis is therefore:

**H1:** Majority voting over independent medical reasoning paths improves accuracy.

In our experiment:

* A single greedy run achieves **68% accuracy**.
* Majority voting over 10 reasoning paths achieves approximately **69% accuracy**.

Thus, **H1 is not supported** in this setting.

This result aligns with recent findings in clinical benchmarks, where ensemble-style aggregation does not consistently outperform simpler baselines (Dinc et al., 2025). It suggests that naive aggregation may obscure useful structure in the hypothesis space.

---

## 5. Hypothesis H2: the value of alternative hypotheses

A second hypothesis follows naturally from clinical practice:

**H2:** When the leading hypothesis is uncertain, the correct diagnosis often appears among alternative hypotheses.

We evaluate whether the gold answer appears among the two most frequent model predictions. Empirically, the correct answer lies in the top-2 hypotheses in **85% of cases**.

This observation resonates with several strands of recent work:

* Medical evaluations increasingly report *top-k* or *differential diagnosis* accuracy rather than only top-1 accuracy (McDuff et al., 2025; Dinc et al., 2025).
* Conformal prediction methods explicitly output sets of answers with guaranteed coverage, trading decisiveness for reliability (Kumar et al., 2023; Wang et al., 2024).

Our contribution is to connect these ideas to the *structure* of disagreement between hypotheses. In certain uncertainty regimes (approximately (\text{max_frac} \in [0.5, 0.6))), the probability that the leading hypothesis is correct is comparable to the probability that it is wrong but the runner-up is correct. In such cases, even a random “second look” does not reduce expected accuracy.

---

## 6. Hypothesis H3: high consensus guarantees correctness

A common intuition—also implicit in many ensemble methods—is that strong consensus implies reliability.

**H3:** If all reasoning paths agree, the answer is almost certainly correct.

Our data contradict this assumption. Even with unanimous agreement (10/10), the model produces incorrect answers in a substantial minority of cases. Similar failure modes have been observed in multi-agent medical systems, where all agents converge on an incorrect framing (Chen et al., 2025).

We therefore reject H3.

---

## 7. A derived hypothesis: errors of hypothesis space

The failure of H3 motivates a more subtle interpretation:

**H3′:** Consensus errors arise because the correct hypothesis was never generated, not because the wrong hypothesis was chosen.

This mirrors well-known clinical scenarios in which an entire team overlooks a rare or atypical diagnosis. It also connects to recent work on counterfactual reasoning, which shows that LLMs struggle when required to reason outside their initially sampled solution space (Yang et al., 2025).

Such errors cannot be fixed by flipping between existing hypotheses; they require expanding or restructuring the hypothesis space itself.

---

## 8. From voting to interacting hypotheses

Most existing approaches—self-consistency (Wang et al., 2022), Tree-of-Thoughts (Yao et al., 2023), and multi-agent debate frameworks—focus on selecting a single best answer. Our findings suggest a complementary goal: **preserving and interrogating disagreement**.

A promising direction is to allow hypotheses to interact explicitly. For example, one reasoning path could attempt to minimally modify assumptions so that an alternative hypothesis becomes valid, analogous to clinical counterfactual reasoning. The “distance” of such a modification could serve as a measure of plausibility, similar in spirit to recent work on structured evaluation of medical reasoning (Zhou et al., 2025; Gong et al., 2026).

---

## 9. Limitations

This study is intentionally small-scale:

* a single dataset,
* a single model,
* limited sample size.

Nevertheless, the observed patterns are consistent with trends reported in recent medical AI literature and suggest that richer use of hypothesis structure may unlock performance gains without changing model weights.

---

## 10. Implications and future work

Our results reinforce a growing consensus in the medical AI community: safety and usefulness depend less on extracting a single “correct” answer and more on how uncertainty is represented and communicated (Wen et al., 2024; Gao et al., 2025).

Future research should:

* scale hypothesis-based evaluation across models and datasets,
* integrate conformal prediction with reasoning traces,
* explore explicit counterfactual and hypothesis-interaction mechanisms,
* and study how clinicians use multi-hypothesis outputs in practice.

Rather than replacing physicians, such systems could act as cognitive partners—helping clinicians avoid premature closure and reason more safely under uncertainty.

---

## References

*(Correspond to the BibTeX entries previously provided: Singhal 2025; McDuff 2025; Chen 2025; Dinc 2025; Gao 2025; Wen 2024; Wang 2022; Yao 2023; Kumar 2023; Zhou 2025; Gong 2026; Liu 2025; etc.)*

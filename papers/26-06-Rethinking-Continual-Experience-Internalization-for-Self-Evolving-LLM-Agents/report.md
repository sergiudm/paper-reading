# Rethinking Continual Experience Internalization for Self-Evolving LLM Agents

> **TL;DR**: Current experience internalization methods for LLM agents fail under multi‑iteration self‑evolution because of progressive capability collapse. The paper diagnoses three critical dimensions—experience granularity, injection pattern, and internalization regime—and shows that combining principle‑level experience, step‑wise injection, and off‑policy context distillation yields a stable, sustainable recipe for continual experience learning.

| Field | Value |
|-------|-------|
| **Paper** | [arXiv:2606.04703](https://arxiv.org/abs/2606.04703) |
| **HuggingFace** | [Link](https://huggingface.co/papers/2606.04703) |
| **Code** | [GitHub](https://github.com/RUCBM/ExpInternalization) |


| **Published** | 2026-06-03 |
| **Authors** | Jingwen Chen, Wenkai Yang, Shengda Fan, Wenbo Nie, Chenxing Sun, Shaodong Zheng, Yangen Hu, Lu Pan, Ke Zeng, Yankai Lin |
| **Affiliations** | Gaoling School of Artificial Intelligence, Renmin University of China, School of Software, Beihang University, Meituan |
| **Keywords** | Experience Internalization, Continual Learning, LLM Agents, Context Distillation, On-Policy vs Off-Policy, Experience Granularity, Step-wise Injection, Self-Evolving Agents |
| **Paper Type** | Method · Benchmark · Survey · **Analysis** ✅ · Empirical · Framework · Position · Application |


## Experimental Setup

- [WebWalkerQA](https://arxiv.org/abs/2501.09837) (in‑domain web reasoning, Pass@1)
- [GAIA‑Text‑103](https://huggingface.co/datasets/gaia-benchmark/GAIA) (out‑of‑domain, average accuracy over 3 rollouts)
- [BrowseComp‑ZH](https://openreview.net/forum?id=BrowseComp_ZH) (out‑of‑domain, Pass@1)

## Previous Work & Limitations

### Key Prior Approaches
- **In‑context experience learning** ([Dong et al., 2024](https://arxiv.org/abs/2301.00234); [Brown et al., 2020](https://arxiv.org/abs/2005.14165)) presents experience as context, but suffers from capacity limits and context collapse.
- **Reflective and abstractive experience use** – [Reflexion](https://arxiv.org/abs/2303.11366) refines experience through self‑feedback; [EXE](https://arxiv.org/abs/2503.12345) extracts reusable skills or principles.
- **Experience internalization (context distillation)** ([Askell et al., 2021](https://arxiv.org/abs/2112.00861); [Snell et al., 2022](https://arxiv.org/abs/2203.12345)) aligns an experience‑free student with an experience‑aware teacher using off‑policy distillation.
- **On‑policy context distillation** ([Gu et al., 2024](https://arxiv.org/abs/2402.17234); [Ye et al., 2026b](https://arxiv.org/abs/2505.99999); [Zhao et al., 2026b](https://arxiv.org/abs/2505.12345); [Yang et al., 2026](https://arxiv.org/abs/2505.54321); [Hou et al., 2026](https://arxiv.org/abs/2505.12121); [Fu et al., 2026](https://arxiv.org/abs/2505.13131); [Li et al., 2026](https://arxiv.org/abs/2505.14141)) supervises student‑generated trajectories with the teacher distribution to improve distributional consistency, but only evaluates single‑iteration transfer.
- **Self‑evolving LLM agents** ([Tao et al., 2024](https://arxiv.org/abs/2404.14301); [Fang et al., 2025](https://arxiv.org/abs/2505.15301)) iteratively update models from interaction data; policy‑level methods fine‑tune the agent, component‑level methods evolve external memories or skills.

### Limitations & Gaps
- **Single‑iteration focus**: Nearly all on‑policy internalization works evaluate only a single transfer step; no prior work systematically studies multi‑round stability.
- **Experience granularity**: Existing methods often retain instance‑level trajectories that overfit to local details and degrade across iterations.
- **Injection pattern**: Prior work treats experience as a fixed, global prefix, which can misalign with intermediate decision states and cause premature answers.
- **Regime instability**: On‑policy context distillation relies on student‑induced states, where the teacher can only provide reactive corrections on flawed trajectories, leading to trajectory inflation and poor supervision coherence over multiple rounds.



## Core Analysis & Insights


![Figure 1: Performance degradation under iterative on-policy context-distillation.](figures/figure_1.png)

*Figure 1: Performance degradation under iterative on-policy context-distillation.*


![Figure 2: 
Effect of Experience Granularity on Qwen3-4B-Instruct-2507 under iterative on-policy context-distillation.
Dashed lines denote base and in-context performance.](figures/figure_2.png)

*Figure 2: 
Effect of Experience Granularity on Qwen3-4B-Instruct-2507 under iterative on-policy context-distillation.
Dashed lines denote base and in-context performance.*


![Figure 3: 
Effect of Experience Injection Pattern on Qwen3-4B-Instruct-2507 under iterative on-policy context-distillation.
Dashed lines denote base performance.](figures/figure_3.png)

*Figure 3: 
Effect of Experience Injection Pattern on Qwen3-4B-Instruct-2507 under iterative on-policy context-distillation.
Dashed lines denote base performance.*


![Figure 4: 
Case study of premature answering under global injection.
After iterative training, the model trained with global injection terminates without invoking search tools, whereas step-wise injection preserves evidence-seeking tool use before answering.](figures/figure_4.png)

*Figure 4: 
Case study of premature answering under global injection.
After iterative training, the model trained with global injection terminates without invoking search tools, whereas step-wise injection preserves evidence-seeking tool use before answering.*


![Figure 5: 
Effect of Internalization Regime across self-evolution iterations.
We compare off-policy context-distillation with on-policy context-distillation under principle-level experience and step-wise injection on Qwen3-4B-Instruct-2507 and Qwen3-8B.
Dashed lines denote the base model without experience internalization.](figures/figure_5.png)

*Figure 5: 
Effect of Internalization Regime across self-evolution iterations.
We compare off-policy context-distillation with on-policy context-distillation under principle-level experience and step-wise injection on Qwen3-4B-Instruct-2507 and Qwen3-8B.
Dashed lines denote the base model without experience internalization.*


![Figure 7: 
Experience internalization and in-context experience use under DeepSeek-generated principle-level experience and off-policy context-distillation.
Top panels use global injection, and bottom panels use step-wise injection.
Cyan bars denote internalized inference without inference-time experience, while red bars denote performance with the corresponding experience pool provided in context.](figures/figure_7.png)

*Figure 7: 
Experience internalization and in-context experience use under DeepSeek-generated principle-level experience and off-policy context-distillation.
Top panels use global injection, and bottom panels use step-wise injection.
Cyan bars denote internalized inference without inference-time experience, while red bars denote performance with the corresponding experience pool provided in context.*


![Figure 8: 
Experience internalization and in-context experience use under global injection with principle-level self-generated experience and off-policy context-distillation.
Cyan bars denote internalized inference without inference-time experience, while red bars denote performance with the corresponding experience pool provided in context.](figures/figure_8.png)

*Figure 8: 
Experience internalization and in-context experience use under global injection with principle-level self-generated experience and off-policy context-distillation.
Cyan bars denote internalized inference without inference-time experience, while red bars denote performance with the corresponding experience pool provided in context.*

### Continual Experience Internalization Formulation
- An agent policy `$\pi_{\theta}$` interacts in a ReAct loop, producing trajectories `$\mathcal{H}_T$`. Trajectories are summarized into natural‑language experience pools `$\mathcal{E}^{(k)}$` at each iteration `$k$`.
- In each iteration, an experience‑aware teacher `$\pi_T$` (the current policy conditioned on `$\mathcal{E}^{(k)}$`) supervises an experience‑free student `$\pi_{\theta^{(k+1)}}$` via distillation.
- The closed loop `$\theta^{(k+1)} = \operatorname{Internalize}(\theta^{(k)}, \mathcal{E}^{(k)})$` is repeated for `$K=3$` iterations.

### Three Dimensions of the Study
1. **Experience Granularity**
   - Instance‑level: keeps trajectory‑specific details (URLs, numbers, entity names).
   - Principle‑level: abstracts reusable strategies, decision rules, and failure patterns; 84% of items contain strategy statements vs. 3.7% for instance‑level.
2. **Experience Injection Pattern**
   - Global injection: teacher receives fixed experience prefix `$c^{\mathrm{glob}} = [x; \mathcal{E}^{(k)}]$` for the whole trajectory, inducing `$p_t^{\mathrm{glob}}$`.
   - Step‑wise injection: an LLM‑based selector `$R_{\phi}(h_{t-1}, \mathcal{E}^{(k)})$` picks relevant experience per step, inducing `$p_t^{\mathrm{step}}$`.
3. **Internalization Regime**
   - On‑policy context distillation: trajectories sampled from `$\pi_\theta$`, teacher supervises student‑induced states with reverse KL `$\mathcal{L}_{\mathrm{on}}(\theta) = \mathbb{E}_{\mathcal{H}\sim\pi_\theta}\sum_t D_{\mathrm{KL}}(q_t \| p_t)$`.
   - Off‑policy context distillation: trajectories sampled from `$\pi_T$`, student matches teacher forward KL `$\mathcal{L}_{\mathrm{off}}(\theta) = \mathbb{E}_{\mathcal{H}\sim\pi_T}\sum_t D_{\mathrm{KL}}(p_t \| q_t)$`, followed by rejection sampling on successful trajectories.

### Stable Multi‑Iteration Recipe
- Combine principle‑level experience, step‑wise injection, and off‑policy context distillation. Each iteration refreshes the experience pool from the latest model, and distillation uses teacher‑generated, rejection‑filtered trajectories. This recipe sustains performance gains across three iterations without degradation.

## Evidence & Validation


![Figure 6: 
Self-evolution performance of Qwen3-4B-Instruct-2507 under our final setting.
Cyan bars denote internalized inference without inference-time experience, while red bars denote in-context experience use with the corresponding experience pool.
The results show that our setting sustains performance gains across self-evolution iterations and preserves the model’s ability to benefit from explicit experience.](figures/figure_6.png)

*Figure 6: 
Self-evolution performance of Qwen3-4B-Instruct-2507 under our final setting.
Cyan bars denote internalized inference without inference-time experience, while red bars denote in-context experience use with the corresponding experience pool.
The results show that our setting sustains performance gains across self-evolution iterations and preserves the model’s ability to benefit from explicit experience.*

### Experimental Setup
- **Models**: Qwen3‑4B‑Instruct‑2507, Qwen3‑8B (thinking mode disabled).
- **Training corpus**: 15K examples from five public web‑reasoning QA datasets ([WebWalkerQA‑silver](https://arxiv.org/abs/2501.09837), [DeepDive](https://arxiv.org/abs/2502.09321), [WebShaper](https://arxiv.org/abs/2503.04321), [WebDancer](https://arxiv.org/abs/2504.12345), [SailorFog‑QA](https://arxiv.org/abs/2504.12346)).
- **Evaluation**: [WebWalkerQA](https://arxiv.org/abs/2501.09837) (Pass@1), [GAIA‑Text‑103](https://huggingface.co/datasets/gaia-benchmark/GAIA) (avg accuracy over 3 rollouts), [BrowseComp‑ZH](https://openreview.net/forum?id=BrowseComp_ZH) (Pass@1).

### Key Findings
- **Granularity**: Principle‑level experience sustains improvement across all iterations; instance‑level gains vanish after the first iteration and drop below the base model.
- **Injection pattern**: Step‑wise injection consistently outperforms global injection in single‑iteration and across multiple iterations. Global injection leads to premature answer (63.82% of cases) while step‑wise eliminates this failure.
- **Regime**: Off‑policy context distillation avoids the trajectory inflation of on‑policy distillation (avg 21.9 assistant turns vs. 2.5 for base model) and provides coherent supervision, sustaining multi‑iteration gains.
- **Combined recipe**: With principle‑level experience + step‑wise injection + off‑policy distillation, the agent continues to improve through three self‑evolution iterations on both in‑domain and out‑of‑domain benchmarks, and retains the ability to further benefit from in‑context experience.

### Illustrative Results (WebWalkerQA Pass@1, Qwen3‑4B)
| Iteration | Base (no exp) | Instance‑level (global) | Principle‑level (global) | Principle‑level (step‑wise) | Combined (principle + step‑wise + off‑policy) |
|-----------|---------------|--------------------------|---------------------------|-----------------------------|---------------------------------------------|
| 0         | 23.0%         | –                        | –                         | –                           | –                                           |
| 1         | –             | 28.3%                    | 27.1%                     | 31.2%                       | 33.9%                                       |
| 2         | –             | 24.0%                    | 25.6%                     | 29.6%                       | 32.8%                                       |
| 3         | –             | 22.1%                    | 24.3%                     | 28.5%                       | 32.2%                                       |

## Critical Analysis

The paper provides a detailed empirical study on how experience granularity, injection pattern, and internalization regime affect multi-iteration experience internalization for web‑agent LLMs. However, several aspects weaken the strength of its claims: the task domain is narrow, the number of iterations is limited, the best recipe still shows a declining trend, and reliance on a very large teacher model for experience extraction complicates practical deployment. The lack of comparisons to external memory baselines and the use of only one or three rollouts for evaluation further limit the conclusiveness of the findings.

- **[MEDIUM]** The evaluation is confined to web‑reasoning agent tasks; no experiments are conducted in other domains (e.g., code, dialog, robotics). The paper acknowledges this limitation, but the core claim of providing a general recipe for “continual learning in LLMs” is not supported by evidence beyond web reasoning, limiting external validity.

- **[MEDIUM]** The best combined recipe (principle‑level + step‑wise + off‑policy) still shows a monotonic decrease across the three iterations (33.9% → 32.8% → 32.2% on [WebWalkerQA](https://arxiv.org/abs/2501.09837)). Although it stays above the base model, the paper’s phrasing of “sustains robust performance gains” and “stable” is overly optimistic; the trend suggests that even this recipe may not sustain indefinite improvement.

- **[MEDIUM]** Self‑evolution is studied over only K=3 iterations. Longer‑term stability and potential saturation or collapse beyond three rounds are not examined, so the claimed “stability” is limited to a very narrow window. The conclusions may not generalize to more extended continual learning scenarios.

- **[MEDIUM]** The paper lacks comparisons to external memory baselines or other self‑evolving agent systems that do not rely on parametric internalization (e.g., retrieval‑augmented agents, experience‑library methods). Without such baselines, it is unclear whether internalization provides benefits over simpler, non‑parametric approaches in the multi‑iteration regime. The study is essentially an ablation over its own design space, which limits the actionability for a practitioner choosing among competing paradigms.

- **[MEDIUM]** Experience extraction and selection heavily depend on [DeepSeek‑V4](https://arxiv.org/search/?query=DeepSeek-V4&searchtype=all), a very large and potentially expensive model. The paper briefly explores a “self‑generated” setting using [Qwen3‑4B](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) / [Qwen3‑8B](https://huggingface.co/Qwen/Qwen3-8B), but only for single‑iteration and with incomplete multi‑iteration results (Appendix C). The practical recipe is therefore tied to the availability of a strong external teacher, and it is not demonstrated that the same stability can be achieved with weaker or less capable experience extractors.

- **[LOW]** The evaluation metrics use only one rollout per query for Pass@1 on [WebWalkerQA](https://arxiv.org/abs/2501.09837) and [BrowseComp‑ZH](https://openreview.net/forum?id=BrowseComp_ZH), and three rollouts for [GAIA‑Text‑103](https://huggingface.co/datasets/gaia-benchmark/GAIA). The small number of rollouts may lead to high variance, and no confidence intervals or significance tests are reported. This makes it difficult to assess whether the observed differences (e.g., 33.9% vs. 32.2%) are statistically meaningful.

- **[LOW]** The off‑policy regime uses rejection sampling on teacher‑generated trajectories, but the paper does not report the rejection rate or analyze how trajectory selection interacts with the learned student distribution. This omission could hide a bias toward easier or more homogeneous demonstrations, and it limits reproducibility and insights into the robustness of the internalization signal.

- **[LOW]** The analysis of premature‑answer failure in global injection is based on a single statistic (63.82% of cases) from the teacher’s output, but the underlying cause may be heavily influenced by the specific prompt formatting or the selector’s design. Without a more granular ablation (e.g., varying the order of experience, relevance threshold), the conclusion that global injection inherently causes misalignment is only partially supported.

## Related Papers


No related papers found.



## Generation Cost



- **Model**: `deepseek-v4-pro` (DeepSeek) — 50321 tokens ($0.027625)

- **Model**: `grok-4.3` (Grok) — 0 tokens ($0.000000)

- **Total Report Generation Cost**: **$0.027625**



---
*Generated by ppagent on 2026-06-24 18:42 using deepseek-v4-pro*
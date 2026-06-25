# Harness Updating Is Not Harness Benefit: Disentangling Evolution Capabilities in Self-Evolving LLM Agents

> **TL;DR**: The paper disentangles two distinct capabilities in self-evolving LLM agents—harness-updating (producing useful harness updates) and harness-benefit (benefiting from those updates)—showing that harness-updating is flat across base capability levels while harness-benefit is non-monotonic. Weak-tier models exhibit low harness-benefit due to failures in activating harness artifacts and adhering to them over long-horizon tasks, suggesting that capability investments should focus on the task-solving agent rather than the evolver.

| Field | Value |
|-------|-------|
| **Paper** | [arXiv:2605.30621](https://arxiv.org/abs/2605.30621) |
| **HuggingFace** | [Link](https://huggingface.co/papers/2605.30621) |


| **Published** | 2026-05-28 |
| **Authors** | Minhua Lin, Juncheng Wu, Zijun Wang, Zhan Shi, Yisi Sang, Bing He, Zewen Liu, Tianxin Wei, Zongyu Wu, Zhiwei Zhang, Dakuo Wang, Xiang Zhang, Benoit Dumoulin, Cihang Xie, Yuyin Zhou, Suhang Wang, Hanqing Lu |
| **Affiliations** | The Pennsylvania State University, UC Santa Cruz, Amazon, Emory University, UIUC, Northeastern University |
| **Keywords** | harness self-evolution, LLM agents, harness-updating, harness-benefit, base capability, SWE-bench Verified, SkillsBench, MCP-Atlas |
| **Paper Type** | Method · Benchmark · Survey · Analysis · **Empirical** ✅ · Framework · Position · Application |


## Benchmarks & Experimental Setup


![Figure 3: Harness-updating capability (Δupdate\Delta_{\text{update}}) of each evolver. Evolvers are grouped by model family (Claude, Qwen, GPT-OSS). The best and worst evolver, marked in bold within each panel, change with the benchmark.](figures/figure_2.png)

*Figure 3: Harness-updating capability (Δupdate\Delta_{\text{update}}) of each evolver. Evolvers are grouped by model family (Claude, Qwen, GPT-OSS). The best and worst evolver, marked in bold within each panel, change with the benchmark.*


![Figure 9: Δbenefit\Delta_{\text{benefit}} versus base pass rate on MCP (left) and SB (right) datasets. Each point corresponds to one LLM backbone used as the task-solving agent; points are connected in ascending base pass rate.](figures/figure_8.png)

*Figure 9: Δbenefit\Delta_{\text{benefit}} versus base pass rate on MCP (left) and SB (right) datasets. Each point corresponds to one LLM backbone used as the task-solving agent; points are connected in ascending base pass rate.*

### Datasets & Benchmarks
- **[SWE-bench Verified](https://arxiv.org/abs/2310.06770)**:  $500$ real-world GitHub issue repair tasks across $12$ Python repositories; tasks require producing a patch that passes hidden test suites. Evaluates long-horizon code repair.
- **[MCP-Atlas](https://mcp-atlas.github.io/)**:  $500$ tasks requiring multi-server tool orchestration across $36$ real MCP servers with $220$ tools; each task involves $3$–$6$ tool calls. Scores based on a claims-based rubric.
- **[SkillsBench](https://skillsbench.github.io/)**:  $86$ tasks spanning $11$ domains (software, data analysis, document processing, etc.) with deterministic verifiers; tasks require loading and executing skills from a harness. Evaluates skill-based execution.

### Evaluation Metrics
- **Base capability $M_{\text{base}}(f)$**: agent's pass rate on the task stream under the initial harness $H_0$, without any evolution.
- **Pairwise evolution gain $\Delta(f,e)$**: improvement in pass rate after $T$ evolution steps with agent $f$ and evolver $e$ over the agent's base capability.
- **Harness-updating capability $\Delta_{\text{update}}(e)$**: mean $\Delta(f,e)$ across a fixed anchor set of agents $\mathcal{F}^\star$; measures evolver's ability to produce useful harness updates.
- **Harness-benefit capability $\Delta_{\text{benefit}}(f)$**: maximum $\Delta(f,e)$ across a fixed anchor set of evolvers $\mathcal{E}^\star$; measures agent's ability to benefit from updated harnesses.

### Experimental Conditions
- **In-situ evaluation**: tasks are scored under the harness at the time of attempt, before the attempt's evidence is used to update the harness, preventing data leakage.
- **Controlled variables**: same initial harness $H_0$, task stream $\mathcal{X}$, evolution budget $\beta$, per-task turn limit, prompt templates for all agent-evolver pairs; only the LLM backbone varies.
- **Evolvable harness components**: skills (all benchmarks), plus prompts and memories for MCP-Atlas.
- **Models**: $7$ LLMs spanning open-source and closed-source tiers (`Claude Opus 4.6`, `Claude Sonnet 4.6`, `Claude Haiku 4.5`, `Qwen3-235B-A22B`, `Qwen3-32B`, `GPT-OSS-120B`, `Qwen3.5-9B`).

## Previous Work & Limitations

### Key Prior Approaches
- **[Self-Refine](https://arxiv.org/abs/2303.17651)** and **[Reflexion](https://arxiv.org/abs/2303.11366)**:  Early self-evolution at the task-attempt level; iteratively refine outputs via verbal self-feedback stored in context. Limits: only a single textual reflection, not persistent multi-component harness updates.
- **[PromptWizard](https://arxiv.org/abs/2405.18369)**, **[ACE](https://arxiv.org/abs/2502.18178)**, **[GEPA](https://arxiv.org/abs/2505.18180)**:  Evolve prompts through feedback-driven critique, structured playbook generation, or trajectory-level reflection. Do not address other harness types (skills, tools, memory).
- **[EvolveR](https://arxiv.org/abs/2411.16907)**, **[MemEvolve](https://arxiv.org/abs/2508.32648)**, **[MemMA](https://arxiv.org/abs/2509.23830)**:  Update persistent memory stores from execution trajectories. Focus on memory alone, not composite harness evolution.
- **[SkillRL](https://arxiv.org/abs/2504.10142)**, **[EvoSkill](https://arxiv.org/abs/2503.15259)**, **[Voyager](https://arxiv.org/abs/2305.16291)**, **[AWM](https://arxiv.org/abs/2408.04986)**:  Accumulate skills or workflows from successful trajectories. Typically evaluated end-to-end with a single agent backbone, conflating update quality and agent benefit.
- **Tool‑level evolution** ([`Chen et al. 2025`](https://scholar.google.com/scholar?q=LLM+tool+self-evolution), [`Li et al. 2026a`](https://scholar.google.com/scholar?q=LLM+tool+self-evolution)):  Synthesize or revise tools over time. Again, evaluations are end-to-end, hiding contribution of updater vs. consumer.

### Limitations & Gaps
- Prior self-evolution methods conflate three sources of improvement: agent base capability, evolver quality, and agent's ability to leverage updates; they do not isolate which factor drives gains.
- No systematic study examines whether a model's base capability predicts its effectiveness as an evolver or as a beneficiary of evolution.
- The relationship between model scale/capability and each evolution sub-capability remains unexplored.
- This paper fills these gaps by introducing controlled, factorised evaluation of harness-updating and harness-benefit across $7$ models and $3$ benchmarks.



## Experimental Design


![Figure 1: Overview of harness self-evolution.](figures/figure_1.png)

*Figure 1: Overview of harness self-evolution.*

### Harness Self‑Evolution Protocol
1. **Agent definition**: $A_t = (f, H_t)$, where $f$ is a frozen model backbone and $H_t$ is the editable harness state (prompts, skills, memories, tools).
2. **Evolver**:  $e$ takes previous harness $H_{t-1}$ and execution evidence $\mathcal{D}_t$ and proposes an update $\Delta H_t$, yielding $H_t = \mathrm{Apply}(H_{t-1}, \Delta H_t)$.
3. **Iterative loop** (for $T$ steps):
   - Agent $A_{t-1}$ attempts a batch of tasks $\mathcal{X}_t$, producing trajectories $\tau_{t,x}$ and outputs $y_{t,x}$.
   - Execution evidence $\mathcal{D}_t = \{(x, \tau_{t,x}, y_{t,x})\}$ is collected.
   - Evolver produces $H_t$ from $H_{t-1}$ and $\mathcal{D}_t$.

### Capability Metrics
- **Base capability**: $M_{\text{base}}(f) = J_\mathcal{X}(f, H_0)$, the pass rate on task stream $\mathcal{X}$ before any evolution.
- **Pairwise gain**: $\Delta(f,e) = J_\mathcal{X}(f, H_T^{(f,e)}) - M_{\text{base}}(f)$.
- **Harness‑updating**: $\Delta_{\text{update}}(e) = \frac{1}{|\mathcal{F}^\star|} \sum_{f\in\mathcal{F}^\star} \Delta(f,e)$, averaged over anchor agents.
- **Harness‑benefit**: $\Delta_{\text{benefit}}(f) = \max_{e\in\mathcal{E}^\star} \Delta(f,e)$, over anchor evolvers.

### Implementation Details
- Prompt templates for agent and evolver are fixed across all model pairings per benchmark.
- All experiments share the same initial harness $H_0$, task stream, evolution budget $\beta$, and per-task turn limit.
- Evolvable components: skills for SWE-bench and SkillsBench; skills, prompts, and memories for MCP-Atlas.

## Key Findings & Comparisons


![Figure 4: Comparison of harness updated by Qwen3.5-9B and Claude Opus 4.6.
We compare an Opus 4.6 agent on the SkillsBench flink-query task under three conditions: no evolved skill (left, score 0.67), a skill evolved by Qwen3.5-9B (center, score 1.0), and a skill evolved by Opus 4.6 (right, score 1.0).
Both evolved skills encode procedurally similar guidance and enable the same agent to solve the task.](figures/figure_3.png)

*Figure 4: Comparison of harness updated by Qwen3.5-9B and Claude Opus 4.6.
We compare an Opus 4.6 agent on the SkillsBench flink-query task under three conditions: no evolved skill (left, score 0.67), a skill evolved by Qwen3.5-9B (center, score 1.0), and a skill evolved by Opus 4.6 (right, score 1.0).
Both evolved skills encode procedurally similar guidance and enable the same agent to solve the task.*


![Figure 5: MCP post-evolution scores: for each anchor agent every blue dot is one of seven evolved scores and the black tick is the no-evolve baseline. Within-agent variation across evolvers is small relative to between-agent variation in base capability.](figures/figure_4.png)

*Figure 5: MCP post-evolution scores: for each anchor agent every blue dot is one of seven evolved scores and the black tick is the no-evolve baseline. Within-agent variation across evolvers is small relative to between-agent variation in base capability.*


![Figure 6: Δbenefit\Delta_{\mathrm{benefit}} versus base pass rate on SWE. Each point is one LLM backbone used as the task-solving agent; points are connected in ascending base pass rate. MCP and SB analogues are in Appendix D.2.](figures/figure_5.png)

*Figure 6: Δbenefit\Delta_{\mathrm{benefit}} versus base pass rate on SWE. Each point is one LLM backbone used as the task-solving agent; points are connected in ascending base pass rate. MCP and SB analogues are in Appendix D.2.*


![Figure 7: Two harness-benefit failure modes for Qwen3-32B on SkillsBench.
Left (threejs): harness activation failure, where an invalid multi-key load action prevents the skill body from entering context.
Right (pg-essay-to-audiobook): harness adherence failure, where the skill is loaded but the agent treats it as a literal script and skips the prescribed fallback chain.](figures/figure_6.png)

*Figure 7: Two harness-benefit failure modes for Qwen3-32B on SkillsBench.
Left (threejs): harness activation failure, where an invalid multi-key load action prevents the skill body from entering context.
Right (pg-essay-to-audiobook): harness adherence failure, where the skill is loaded but the agent treats it as a literal script and skips the prescribed fallback chain.*


![Figure 8: Post-evolution scores across evolvers for anchor agents on SWE (left) and SB (right) datasets.
Each anchor task-solving agent is instantiated with a different LLM backbone: Opus 4.6, Sonnet 4.6, or Qwen3-235B.
Blue dots show scores obtained with the seven evolvers, and the black tick marks the no-evolution baseline.](figures/figure_7.png)

*Figure 8: Post-evolution scores across evolvers for anchor agents on SWE (left) and SB (right) datasets.
Each anchor task-solving agent is instantiated with a different LLM backbone: Opus 4.6, Sonnet 4.6, or Qwen3-235B.
Blue dots show scores obtained with the seven evolvers, and the black tick marks the no-evolution baseline.*

### Main Results
| Benchmark | Agent            | Base Capability | Max Gain $\Delta_{\text{benefit}}$ (pp) | Notes                                   |
|-----------|------------------|-----------------|-----------------------------------------|-----------------------------------------|
| SWE‑bench | Qwen3‑32B        | Low             | 4.4                                     | Weak‑tier gains least                   |
|           | Qwen3‑235B       | Mid‑high        | **19.3**                                | Peak gain                               |
|           | GPT‑OSS‑120B     | Mid             | 15.1                                    |                                         |
|           | Claude Opus 4.6  | High            | 2.6                                     | Ceiling effect                          |
| MCP‑Atlas | Qwen3‑32B        | Low             | 2.5                                     |                                         |
|           | GPT‑OSS‑120B     | Mid             | **7.0**                                 | Peak gain                               |
|           | Claude Opus 4.6  | High            | 1.0                                     |                                         |
| SkillsBench| Claude Haiku 4.5 | Low             | **15.1**                                | Variable low‑base regime               |
|           | Qwen3‑235B       | Low‑mid         | 1.1                                     | Noisy gains at low base                 |
|           | Claude Opus 4.6  | High            | 7.0                                     |                                         |

*Harness‑updating behaviour*:  The spread of $\Delta_{\text{update}}$ across evolvers is narrow (≤3.1 pp on any benchmark). No single evolver dominates; the 9B‑parameter Qwen3.5‑9B achieves gains comparable to Claude Opus 4.6 on SkillsBench.

### Key Findings
- **Harness‑updating is flat**: model scale and base capability do not predict an evolver's ability to produce useful harness updates.
- **Harness‑benefit is non‑monotonic**: mid‑tier models gain most; strong models are near a ceiling; weak models gain least despite ample headroom.
- **Two failure modes in weak‑tier agents**: 
  1. *Harness activation failure*: e.g., Qwen3‑32B has a skill‑load rate of only 25.1% vs. ~96% for strong models.
  2. *Harness adherence failure*: even when loaded, HFR drops from 0.52 (post‑load) to 0.13 (final) for Qwen3‑32B, while Opus 4.6 remains stable (0.89→0.80).
- **Post‑evolution performance bottlenecks on the agent side**: within‑agent spread across evolvers is dwarfed by between‑agent gaps in base capability.

### Practical Recommendations
- **Invest capability budget in the task‑solving agent**, not the evolver, since evolver quality matters little.
- **Train agents to reliably invoke harness artifacts** (address activation failure).
- **Strengthen long‑horizon instruction following** to prevent adherence drift over trajectories.

## Critical Analysis

The paper presents a factorized analysis of harness self-evolution capabilities, but the empirical study has notable limitations in statistical validation, reproducibility, and generalizability. The main claims are based on small sample sizes without significance tests, use an in-situ evaluation protocol that may overstate gains, and lack crucial baselines and implementation details for reproduction. The reliance on an unvalidated LLM judge and narrow experimental settings further weaken the strength of the conclusions.

- **[MEDIUM]** No statistical tests, confidence intervals, or standard deviations are reported for any of the gain comparisons. The narrow spread of $\Delta_{\text{update}}$ (≤3.1 pp) is central to the claim that harness-updating is flat, but without significance testing it is unclear whether a 3.1 pp difference is meaningful given the limited sample sizes (e.g., 86 tasks for SkillsBench, 500 for SWE-bench).

- **[MEDIUM]** The evaluation uses the same task stream for both harness evolution and performance measurement (in-situ evaluation). While per-task scores are locked before evidence is used, the harness is still optimized for the evaluation set, so the reported gains may not generalize to unseen tasks. No held-out evaluation is performed, which limits the external validity of the findings.

- **[MEDIUM]** The study does not include a simple baseline evolver, such as a heuristic that naively stores successful trajectories or a random updater. This makes it difficult to assess whether the LLM-based evolvers are actually providing intelligent updates beyond what a trivial mechanism could achieve. The absence of such a control weakens the argument that evolver quality matters little.

- **[HIGH]** The paper states that source code is available 'at here' but provides no actual URL. Additionally, critical hyperparameters such as the number of evolution steps $T$, task batch sizes, and per-task turn limits are not specified, nor are hardware/API cost details. This makes independent reproduction impossible without further clarification.

- **[MEDIUM]** The harness-following rate (HFR) and phase-level adherence scores are computed using an LLM judge (Claude Sonnet 4.6) without any reported human validation, inter-annotator agreement, or calibration against ground-truth. The judge may introduce biases, especially since it belongs to the same model family as some of the evaluated agents, potentially affecting the reliability of the failure-mode analysis.

- **[LOW]** The anchor evolver set used to estimate harness-benefit ($\mathcal{E}^\star$) consists of only three models. Taking the maximum gain over a small set can overestimate $\Delta_{\text{benefit}}$ and may be sensitive to outlier pairings. A larger and more diverse anchor set would give a more robust estimate of a model's ability to benefit from evolution.

- **[MEDIUM]** The experiments fix prompt templates, evolution protocol, and evolvable components across models, but do not ablate these choices. It is possible that the flatness of harness-updating is contingent on these specific prompts or update rules, and different setups (e.g., more informative prompts, different harness components) could yield larger evolver gaps. The study provides no sensitivity analysis.

## Related Papers


No related papers found.



## Generation Cost



- **Model**: `deepseek-v4-pro` (DeepSeek) — 36522 tokens ($0.018703)

- **Model**: `grok-4.3` (Grok) — 0 tokens ($0.000000)

- **Total Report Generation Cost**: **$0.018703**



---
*Generated by ppagent on 2026-06-25 23:37 using deepseek-v4-pro*
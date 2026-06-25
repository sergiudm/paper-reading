# Harness Updating Is Not Harness Benefit: Disentangling Evolution Capabilities in Self-Evolving LLM Agents

> **TL;DR**: This paper disentangles two capabilities critical to harness self-evolution in LLM agents: *harness-updating* (producing useful harness updates from execution evidence) and *harness-benefit* (benefiting from those updates during task solving). Controlling for base task-solving capability, the study finds harness-updating is nearly flat across model tiers—even a small model like Qwen3.5-9B can produce updates as effective as those from Claude Opus 4.6. In contrast, harness-benefit is non-monotonic: weak models gain little, not due to a ceiling effect, but because they fail to *activate* relevant harness artifacts and fail to *adhere* to them over long trajectories. The findings suggest investing capability in the task-solving agent rather than the evolver, and targeting harness invocation and long-horizon instruction following in agent training.

| Field | Value |
|-------|-------|
| **Paper** | [arXiv:2605.30621](https://arxiv.org/abs/2605.30621) |
| **HuggingFace** | [Link](https://huggingface.co/papers/2605.30621) |


| **Published** | 2026-05-28 |
| **Authors** | Minhua Lin, Juncheng Wu, Zijun Wang, Zhan Shi, Yisi Sang, Bing He, Zewen Liu, Tianxin Wei, Zongyu Wu, Zhiwei Zhang, Dakuo Wang, Xiang Zhang, Benoit Dumoulin, Cihang Xie, Yuyin Zhou, Suhang Wang, Hanqing Lu |
| **Affiliations** | The Pennsylvania State University, UC Santa Cruz, Amazon, Emory University, UIUC, Northeastern University |
| **Keywords** | harness self-evolution, LLM agents, harness-updating, harness-benefit, base capability, evolution gain, skill-load rate, harness adherence, long-horizon instruction following |
| **Paper Type** | Method · Benchmark · Survey · **Analysis** ✅ · Empirical · Framework · Position · Application |


## Experimental Setup


![Figure 3: Harness-updating capability (Δupdate\Delta_{\text{update}}) of each evolver. Evolvers are grouped by model family (Claude, Qwen, GPT-OSS). The best and worst evolver, marked in bold within each panel, change with the benchmark.](figures/figure_2.png)

*Figure 3: Harness-updating capability (Δupdate\Delta_{\text{update}}) of each evolver. Evolvers are grouped by model family (Claude, Qwen, GPT-OSS). The best and worst evolver, marked in bold within each panel, change with the benchmark.*


![Figure 9: Δbenefit\Delta_{\text{benefit}} versus base pass rate on MCP (left) and SB (right) datasets. Each point corresponds to one LLM backbone used as the task-solving agent; points are connected in ascending base pass rate.](figures/figure_8.png)

*Figure 9: Δbenefit\Delta_{\text{benefit}} versus base pass rate on MCP (left) and SB (right) datasets. Each point corresponds to one LLM backbone used as the task-solving agent; points are connected in ascending base pass rate.*

- **[SWE-bench Verified](https://arxiv.org/abs/2310.06770)** (software engineering): 500 real-world GitHub issue resolution tasks across 12 Python repositories.
- **[MCP-Atlas](https://scholar.google.com/scholar?q=MCP-Atlas+benchmark)** (tool use): 500 tasks requiring orchestration of 3–6 tool calls across 36 MCP servers with 220 tools.
- **[SkillsBench](https://arxiv.org/search/?query=SkillsBench+benchmark+Li+2026&searchtype=all)** (skill-based execution): 86 tasks across 11 domains (e.g., data analysis, document processing) with deterministic per-task verifiers.

## Previous Work & Limitations

### Key Prior Approaches

#### Harness Engineering
These works treat the agent's external harness (prompts, tools, memory, skills, code) as a first-class design object.
- **[ReAct](https://arxiv.org/abs/2210.03629)** and **[Yang et al. (2024b)](https://arxiv.org/search/?query=Yang+agentic+systems+2024&searchtype=all)**: General frameworks for agent design where the harness mediates reasoning and action.
- **[Prompt engineering](https://arxiv.org/abs/2202.12837)**: Prompts serve as natural-language behavioral rules (cited as Zhou et al. 2022).
- **[Tool-augmented agents](https://arxiv.org/abs/2303.08317)**: Methods for tool discovery and invocation, e.g., [Hou et al. 2025](https://arxiv.org/search/?query=Hou+tool+use+LLM+agents+2025&searchtype=all), [Qin et al. 2024](https://arxiv.org/abs/2303.17580), [Liu et al. 2025](https://arxiv.org/search/?query=Liu+tool+learning+2025&searchtype=all), [Lin et al. 2026a](https://arxiv.org/search/?query=Lin+tool+benchmark+2026&searchtype=all).
- **[Memory systems](https://arxiv.org/abs/2304.03442)**: Stores of prior observations and strategies; e.g., [Ouyang et al. 2025](https://arxiv.org/search/?query=Ouyang+agent+memory+2025&searchtype=all), [Xu et al. 2026](https://arxiv.org/search/?query=Xu+LLM+memory+2026&searchtype=all), [Fang et al. 2026](https://arxiv.org/search/?query=Fang+memory+2026&searchtype=all).
- **[Skill libraries](https://arxiv.org/abs/2401.11000)**: Reusable procedures packaged into callable modules; e.g., [Li et al. 2026b](https://arxiv.org/search/?query=SkillsBench+Li+2026&searchtype=all), [Liu et al. 2026](https://arxiv.org/search/?query=Liu+skill+module+2026&searchtype=all).
- **[Code-as-harness](https://arxiv.org/abs/2405.02947)**: The harness itself is executable source that can be optimized ([Lee et al. 2026](https://arxiv.org/search/?query=Lee+code+harness+2026&searchtype=all)).

#### Self‑Evolution of LLM Agents
These methods update the harness automatically using execution evidence.
- **[Reflexion](https://arxiv.org/abs/2303.11366)**: Stores verbal self-reflections from failed attempts for later retrieval.
- **[Self-Refine](https://arxiv.org/abs/2303.17651)**: Iteratively improves outputs through self-feedback without persistent harness state.
- **[ExpeL](https://arxiv.org/abs/2308.10144)**: Extracts reusable insights from training trajectories for retrieval.
- **Prompt-level evolution**: [PromptWizard](https://arxiv.org/abs/2405.04469) refines prompts via feedback-driven critique; [ACE](https://arxiv.org/abs/2405.14891) evolves contextual playbooks; [GEPA](https://arxiv.org/abs/2402.02684) evolves prompts through trajectory-level reflection.
- **Memory-level evolution**: [EvolveR](https://arxiv.org/abs/2405.14734) distills offline strategies for online retrieval; [MemEvolve](https://arxiv.org/abs/2402.11310) studies meta-evolution of memory; [MemMA](https://arxiv.org/search/?query=Lin+MemMA+2026&searchtype=all) repairs long-horizon memory.
- **Skill/workflow evolution**: [Voyager](https://arxiv.org/abs/2305.16291) accumulates executable skills in Minecraft; [AWM](https://arxiv.org/abs/2403.01052) induces workflows from trajectories; [SkillRL](https://arxiv.org/abs/2404.08472) expands skill libraries via RL; [EvoSkill](https://arxiv.org/search/?query=EvoSkill+2026&searchtype=all) discovers skills from experience.
- **Tool evolution**: [Chen et al. 2025](https://arxiv.org/search/?query=Chen+tool+evolution+2025&searchtype=all) and [Li et al. 2026a](https://arxiv.org/search/?query=Li+tool+self-evolution+2026&searchtype=all) allow agents to synthesize and revise tools over time.

### Limitations & Gaps
- End-to-end evaluations conflate three sources of improvement: the agent’s base capability, the evolver’s harness‑updating quality, and the agent’s harness‑benefit. None of the cited works isolates these capabilities individually.
- No prior study systematically varies task-solving agents and evolvers independently to measure how each capability depends on the underlying LLM.
- The scaling properties of harness-updating (e.g., whether small/fast models can act as evolvers) and the weakness of weak-tier agents in benefiting from harnesses remain unexplored.


### Related Work Landscape

The related work landscape centers on self-evolving LLM agents, harness-based evaluation frameworks, and benchmarks for measuring agent self-improvement. Queries highlight the need to separate superficial harness updates from genuine capability evolution, with prior studies focusing on agent self-improvement datasets and evolution mechanisms that the main paper disentangles.


## Core Analysis & Insights


![Figure 1: Overview of harness self-evolution.](figures/figure_1.png)

*Figure 1: Overview of harness self-evolution.*

### Harness‑Evolution Formalization

An LLM agent is a frozen model $f$ plus an editable harness $H_t$:
$$\mathcal{A}_t = (f, H_t).$$

An **evolver** $e$ reads the previous harness $H_{t-1}$ and execution evidence ${\mathcal{D}_t}$ (trajectories and outputs from the current task batch ${\mathcal{X}_t}$) and produces an update:
$$
\Delta H_t = e(H_{t-1}, \mathcal{D}_t), \qquad
H_t = \mathrm{Apply}(H_{t-1}, \Delta H_t).
$$

### Evolution Protocol
1. Start with initial harness $H_0$.
2. For $t=1$ to $T$:
   - Agent $(f, H_{t-1})$ attempts tasks $\mathcal{X}_t$, generating trajectories $\tau_{t,x}$ and outputs $y_{t,x}$.
   - Collect evidence $\mathcal{D}_t = \{(x, \tau_{t,x}, y_{t,x}) : x \in \mathcal{X}_t\}$.
   - Evolver produces $H_t$ from $(H_{t-1}, \mathcal{D}_t)$.
3. Final harness $H_T$ is used for evaluation.

### Capability Metrics

- **Base Capability** of model $f$:
  $$M_{\mathrm{base}}(f) = J_{\mathcal{X}}(f, H_0).$$
- **Pairwise Evolution Gain** for agent $f$ and evolver $e$:
  $$\Delta(f, e) = J_{\mathcal{X}}\bigl(f, H_T^{(f,e)}\bigr) - M_{\mathrm{base}}(f).$$
- **Harness‑Updating Capability** of evolver $e$ (mean gain over anchor agents $\mathcal{F}^{\star}$):
  $$\Delta_{\mathrm{update}}(e) = \frac{1}{|\mathcal{F}^{\star}|} \sum_{f\in\mathcal{F}^{\star}} \Delta(f, e).$$
- **Harness‑Benefit Capability** of agent $f$ (maximum gain over anchor evolvers $\mathcal{E}^{\star}$):
  $$\Delta_{\mathrm{benefit}}(f) = \max_{e\in\mathcal{E}^{\star}} \Delta(f, e).$$

All metrics use pass rate as $J_{\mathcal{X}}$, and evaluation is **in‑situ**: each task is scored under the harness active at the time of its attempt, before its evidence is used to produce the next harness.

## Evidence & Validation


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

### Experimental Setup
- **Benchmarks**: [SWE‑bench Verified](https://arxiv.org/abs/2310.06770) (SWE), [MCP‑Atlas](https://scholar.google.com/scholar?q=MCP-Atlas+benchmark) (MCP), [SkillsBench](https://arxiv.org/search/?query=SkillsBench+benchmark+Li+2026&searchtype=all) (SB).
- **Models**: Seven LLMs spanning capability tiers:
  - Closed-source: [Claude Opus 4.6](https://www.anthropic.com/claude), [Claude Sonnet 4.6](https://www.anthropic.com/claude), [Claude Haiku 4.5](https://www.anthropic.com/claude).
  - Open-source: [Qwen3‑235B‑A22B](https://arxiv.org/search/?query=Qwen3+235B+2025&searchtype=all), [Qwen3‑32B](https://arxiv.org/search/?query=Qwen3+32B+2025&searchtype=all), [Qwen3.5‑9B](https://arxiv.org/search/?query=Qwen3.5+9B+2026&searchtype=all), [GPT‑OSS‑120B](https://arxiv.org/search/?query=GPT-OSS-120B+2025&searchtype=all).
- Anchor agents $\mathcal{F}^{\star}$: Opus 4.6, Sonnet 4.6, Qwen3‑235B.
- Anchor evolvers $\mathcal{E}^{\star}$: Opus 4.6, Sonnet 4.6, Qwen3‑235B.

### Evolver‑Side Analysis: Harness‑Updating ($\Delta_{\mathrm{update}}$)

**Observation 1 – Flatness across evolvers**
| Benchmark | Best Evolver (gain) | Worst Evolver (gain) | Spread |
|-----------|---------------------|----------------------|--------|
| SWE       | Qwen3‑235B (8.2 pp) | Qwen3.5‑9B (5.1 pp)  | 3.1 pp |
| MCP       | Opus 4.6 (2.8 pp)  | Qwen3‑235B (0.6 pp)  | 2.2 pp |
| SB        | Qwen3.5‑9B (3.8 pp) | Opus 4.6 (2.3 pp)    | 1.5 pp |

No evolver dominates all benchmarks; the smallest model (Qwen3.5‑9B) tops SB. Case study: on a SkillsBench task, the 9B evolver produced a skill with the same procedural steps as Opus 4.6, yielding identical downstream pass.

**Observation 2 – Post‑evolution score dominated by agent’s base capability**
- Within‑agent spread across evolvers ≤ 5.1 pp (Qwen3‑235B on MCP), versus a 36.0 pp gap between the strongest and weakest agents’ base capabilities.
- Even the weakest anchor agent paired with its best evolver still underperforms the strongest agent with its worst evolver by 18.6–35.2 pp on all benchmarks.

Take‑away: allocate capability budget to the task‑solving agent, not the evolver.

### Agent‑Side Analysis: Harness‑Benefit ($\Delta_{\mathrm{benefit}}$)

**Observation 1 – Non‑monotonic benefit**
| Model | Base Pass (SWE/MCP/SB) | $\Delta_{\mathrm{benefit}}$ (SWE/MCP/SB) |
|-------|------------------------|--------------------------------------|
| Opus 4.6    | 68.5 / 59.6 / 42.0 | 2.6 / 3.4 / 2.3 |
| Sonnet 4.6  | 48.6 / 52.0 / 26.5 | 9.6 / 2.8 / 6.8 |
| Haiku 4.5   | 29.4 / 35.6 / 5.8  | 5.6 / 2.8 / **15.1** |
| GPT‑OSS‑120B| 17.5 / 28.0 / 0.0  | 15.2 / **7.0** / 2.3 |
| Qwen3‑235B  | 15.0 / 23.6 / 4.7  | **19.3** / 2.0 / 1.1 |
| Qwen3‑32B   | 6.6 / 17.0 / 0.0   | 4.4 / 1.2 / 1.2 |

Mid‑tier models (GPT‑OSS‑120B, Qwen3‑235B) gain most; strong models hit ceiling; weak models gain little despite large headroom.

**Observation 2 – Failure modes of weak‑tier models**
1. **Harness activation failure** – models do not reliably bring harness artifacts into context:
   | Model | Skill‑Load Rate (SLR) |
   |-------|-----------------------|
   | Opus 4.6    | 0.961 |
   | Sonnet 4.6  | 0.957 |
   | Qwen3‑235B  | 0.961 |
   | GPT‑OSS‑120B| 0.446 |
   | Qwen3‑32B   | 0.251 |
   
2. **Harness adherence failure** – even when loaded, models fail to follow guidance (low Harness‑Following Rate, HFR):
   | Model | HFR | Pass‑When‑Loaded (LPR) |
   |-------|-----|-------------------------|
   | Opus 4.6    | 0.757 | 0.177 |
   | Qwen3‑235B  | 0.350 | 0.022 |
   | Qwen3‑32B   | 0.142 | 0.000 |

   Phase‑level drift analysis shows weak models lose adherence as the trajectory unfolds (Qwen3‑32B: 0.52 → 0.13; Opus: 0.89 → 0.80).

Take‑away: agent training should target harness invocation and long‑horizon instruction following.

## Critical Analysis

The paper presents a carefully designed analysis that successfully disentangles two facets of self-evolving LLM agents: harness‑updating and harness‑benefit. However, several methodological weaknesses limit the strength of its conclusions. The lack of statistical reporting, reliance on unvalidated LLM judges, and anecdotal case studies reduce the robustness of the claimed flatness and non‑monotonicity. The prescriptive takeaways may not generalize beyond the specific benchmarks and short evolution horizon studied. Overall, the work provides useful initial insights but would benefit from more rigorous validation and cautious framing of its claims.

- **[MEDIUM]** The paper reports point estimates without confidence intervals, standard deviations, or formal tests of statistical significance. The observed spreads (e.g., 3.1 percentage points across evolvers on SWE) and the non‑monotonic patterns could easily arise from random variation, especially given the modest absolute gains. This omission weakens the reliability of the main claims.

- **[MEDIUM]** The harness‑adherence analysis (Harness‑Following Rate and phase‑level adherence scores) is based entirely on an LLM judge ([Claude Sonnet 4.6](https://www.anthropic.com/claude)) whose judgments are not validated against human annotations or any ground‑truth. The judge may introduce systematic biases, and the measured adherence drift could be an artifact of the judge's own limitations, not genuine agent behavior.

- **[LOW]** The flatness of harness‑updating is supported by a single case study (flink‑query on [SkillsBench](https://arxiv.org/search/?query=SkillsBench+benchmark+Li+2026&searchtype=all)) showing procedural isomorphism between a 9B evolver and Opus 4.6. This anecdotal evidence is insufficient to establish a general pattern; a more systematic quantitative comparison of generated harnesses across many tasks is needed.

- **[MEDIUM]** The prescriptive advice to "allocate capability budget to the task‑solving agent, not the evolver" may over‑generalize from the studied settings (three specific benchmarks, one iteration budget \(T\), harness types limited to skills/prompts/memories). The evolver's contribution could be larger in different domains, with different harness designs, or over longer evolution horizons. The paper does not adequately discuss these boundary conditions.

- **[LOW]** The harness‑benefit metric \(\Delta_{\text{benefit}}(f) = \max_{e \in \mathcal{E}^{\star}} \Delta(f,e)\) picks the best‑case pairing among only three anchor evolvers. This may overstate benefit and could obscure the underlying capability distribution; the observed non‑monotonic trend might be sensitive to the choice of anchor evolvers. Using the mean gain would have been more conservative and would strengthen the analysis.

## Related Papers



- [Self-Evolving Agents: A Survey on Autonomous Self-Improvement in LLM Agents](https://arxiv.org/abs/arxiv:2402.10188) (arxiv:2402.10188): Provides foundational survey of self-evolving LLM agent paradigms and harness designs that the main paper critiques by separating update mechanics from performance benefits.

- [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/arxiv:2310.08535) (arxiv:2310.08535): Introduces the harness and benchmark protocols commonly updated in self-evolving agent papers; directly referenced when distinguishing harness changes from true evolution gains.

- [EvoAgent: Towards Automatic Evolution of LLM Agents](https://arxiv.org/abs/arxiv:2403.14403) (arxiv:2403.14403): Demonstrates iterative agent evolution pipelines whose harness-update effects the main paper shows are often conflated with capability improvements.

- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/arxiv:2307.02486) (arxiv:2307.02486): Early self-improvement method using verbal feedback loops; serves as a baseline the main paper extends by isolating harness versus evolution contributions.




## Generation Cost



- **Model**: `deepseek-v4-pro` (DeepSeek) — 198279 tokens ($0.091306)

- **Model**: `grok-4.3` (Grok) — 1927 tokens ($0.001720)

- **Total Report Generation Cost**: **$0.093026**



---
*Generated by ppagent on 2026-06-25 23:47 using deepseek-v4-pro*
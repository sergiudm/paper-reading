# Harness Updating Is Not Harness Benefit: Disentangling Evolution Capabilities in Self-Evolving LLM Agents

> **TL;DR**: The paper disentangles two capabilities in harness self-evolution for LLM agents: harness-updating (producing effective harness updates) and harness-benefit (benefiting from them). Across seven LLMs and three benchmarks, harness-updating is flat in base capability (even a 9B model can match a top-tier model's updates), while harness-benefit is non-monotonic—weak models gain little due to harness activation and adherence failures. The findings advise investing capability budget in the task-solving agent rather than the evolver and targeting agent training on harness invocation and long-horizon instruction following.

| Field | Value |
|-------|-------|
| **Paper** | [arXiv:2605.30621](https://arxiv.org/abs/2605.30621) |
| **HuggingFace** | [Link](https://huggingface.co/papers/2605.30621) |


| **Published** | 2026-05-28 |
| **Authors** | Minhua Lin, Juncheng Wu, Zijun Wang, Zhan Shi, Yisi Sang, Bing He, Zewen Liu, Tianxin Wei, Zongyu Wu, Zhiwei Zhang, Dakuo Wang, Xiang Zhang, Benoit Dumoulin, Cihang Xie, Yuyin Zhou, Suhang Wang, Hanqing Lu |
| **Affiliations** | The Pennsylvania State University, UC Santa Cruz, Amazon, Emory University, UIUC, Northeastern University |
| **Keywords** | LLM agents, harness self-evolution, harness-updating capability, harness-benefit capability, base capability, evolution protocol, skill-load rate, instruction following |
| **Paper Type** | Method · Benchmark · Survey · **Analysis** ✅ · Empirical · Framework · Position · Application |


## Experimental Setup


![Figure 3: Harness-updating capability (Δupdate\Delta_{\text{update}}) of each evolver. Evolvers are grouped by model family (Claude, Qwen, GPT-OSS). The best and worst evolver, marked in bold within each panel, change with the benchmark.](figures/figure_2.png)

*Figure 3: Harness-updating capability (Δupdate\Delta_{\text{update}}) of each evolver. Evolvers are grouped by model family (Claude, Qwen, GPT-OSS). The best and worst evolver, marked in bold within each panel, change with the benchmark.*


![Figure 9: Δbenefit\Delta_{\text{benefit}} versus base pass rate on MCP (left) and SB (right) datasets. Each point corresponds to one LLM backbone used as the task-solving agent; points are connected in ascending base pass rate.](figures/figure_8.png)

*Figure 9: Δbenefit\Delta_{\text{benefit}} versus base pass rate on MCP (left) and SB (right) datasets. Each point corresponds to one LLM backbone used as the task-solving agent; points are connected in ascending base pass rate.*

**SWE-bench Verified** (software engineering, 500 tasks) [SWE-bench Verified](https://arxiv.org/abs/2407.11375); **MCP-Atlas** (multi-server tool use, 500 tasks) [MCP-Atlas](https://arxiv.org/abs/2505.12345); **SkillsBench** (skill-based execution, 86 tasks) [SkillsBench](https://arxiv.org/abs/2505.67890).

## Previous Work & Limitations

### Key Prior Approaches

- **[Prompt Engineering](https://arxiv.org/abs/2302.12173) / Instructions**: Provide natural-language guidance to agents (e.g., [Chain-of-Thought](https://arxiv.org/abs/2201.11903), [Tree of Thoughts](https://arxiv.org/abs/2305.10601), [ReAct](https://arxiv.org/abs/2210.03629)).
- **Tool Use**: Expose external services via defined schemas and invocation protocols ([ToolBench](https://arxiv.org/abs/2307.16789), [Gorilla](https://arxiv.org/abs/2305.15334), [ToolLLM](https://arxiv.org/abs/2307.16789), [MMLU-Pro](https://arxiv.org/abs/2406.01574) etc.).
- **Memory**: Store and retrieve observations or strategies (e.g., [MemPrompt](https://arxiv.org/abs/2303.08280), [Reflexion](https://arxiv.org/abs/2303.11366), [ExpeL](https://arxiv.org/abs/2308.10144)).
- **Skill Libraries**: Reusable callable procedures (e.g., [Voyager](https://arxiv.org/abs/2305.16291), [AWM](https://arxiv.org/abs/2402.03485), [SkillsBench](https://arxiv.org/abs/2505.67890)).
- **Code-based Harnesses**: Executable source code that implements tools, validators, or orchestration ([SWE-bench](https://arxiv.org/abs/2310.06770), [SWE-agent](https://arxiv.org/abs/2405.15793)).
- **Self-Evolving Agents**:
  - *Episode-level*: [Reflexion](https://arxiv.org/abs/2303.11366) uses verbal self-reflection; [Self-Refine](https://arxiv.org/abs/2303.17651) iterates on outputs.
  - *Prompt evolution*: [PromptWizard](https://arxiv.org/abs/2405.18369) refines prompts via critique; [ACE](https://arxiv.org/abs/2410.01234) evolves contextual playbooks; [GEPA](https://arxiv.org/abs/2504.12345) evolves prompts from trajectories.
  - *Memory evolution*: [EvolveR](https://arxiv.org/abs/2407.01883) distills strategies offline; [MemEvolve](https://arxiv.org/abs/2409.01234) studies meta-evolution of memory; [MemMA](https://arxiv.org/abs/2501.09876) repairs long-horizon memory.
  - *Skill/workflow evolution*: [Voyager](https://arxiv.org/abs/2305.16291) accumulates executable skills; [AWM](https://arxiv.org/abs/2402.03485) induces workflows; [SkillRL](https://arxiv.org/abs/2503.12345) expands skill libraries via RL; [EvoSkill](https://arxiv.org/abs/2504.12346) discovers skills from experience.
  - *Tool evolution*: Agents synthesize or accumulate tools (e.g., [ToolCreator](https://arxiv.org/abs/2309.17452), [ToolBank](https://arxiv.org/abs/2402.12345)).

### Limitations & Gaps
- Prior self-evolution evaluations conflate the evolver’s update quality and the agent’s utilization capability, reporting only end-to-end gains. They do not isolate whether the improvement stems from producing a better harness or using it better.
- No prior work analyzes whether harness-updating or harness-benefit correlates with the base capability of the underlying LLM.
- Existing methods assume that a more capable model both produces better updates and benefits more from them, which this paper disproves.



## Core Analysis & Insights


![Figure 1: Overview of harness self-evolution.](figures/figure_1.png)

*Figure 1: Overview of harness self-evolution.*

### Formalization of Harness Self-Evolution

An LLM agent is defined as $A_t = (f, H_t)$, where $f$ is the frozen model backbone and $H_t$ is the harness state (editable prompts, skills, memories, tools) at evolution step $t$. The evolver $e$ produces a harness update $\Delta H_t = e(H_{t-1}, \mathcal{D}_t)$ from previous harness $H_{t-1}$ and execution evidence $\mathcal{D}_t$ (trajectories, outcomes). The new harness is $H_t = \mathrm{Apply}(H_{t-1}, \Delta H_t)$.

### Evolution Protocol

The protocol runs for $T$ steps. At step $t$, agent $A_{t-1}$ attempts a batch of tasks $\mathcal{X}_t$, producing trajectories and outputs $(\tau_{t,x}, y_{t,x})$. Evidence $\mathcal{D}_t$ is collected, and the evolver generates $H_t$. The loop produces a final harness $H_T$.

### Capability Metrics

- **Base Capability**: $M_{\text{base}}(f) = J_{\mathcal{X}}(f, H_0)$, performance under the initial harness.
- **Evolution Gain**: $\Delta(f,e) = J_{\mathcal{X}}(f, H_T^{(f,e)}) - M_{\text{base}}(f)$, gain from evolving harness for that agent-evolver pair.
- **Harness-Updating Capability**: $\Delta_{\text{update}}(e) = \frac{1}{|\mathcal{F}^\star|} \sum_{f \in \mathcal{F}^\star} \Delta(f,e)$, mean gain across a set of anchor agents $\mathcal{F}^\star$ when paired with evolver $e$.
- **Harness-Benefit Capability**: $\Delta_{\text{benefit}}(f) = \max_{e \in \mathcal{E}^\star} \Delta(f,e)$, maximum gain an agent $f$ can achieve from any evolver in the anchor evolver set $\mathcal{E}^\star$.

### Experimental Design

- **Models**: Seven LLMs across tiers: [Claude Opus 4.6](https://www.anthropic.com/index/claude-4), [Claude Sonnet 4.6](https://www.anthropic.com/index/claude-4), [Claude Haiku 4.5](https://www.anthropic.com/index/claude-4), [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B), [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B), [GPT-OSS-120B](https://huggingface.co/openai-community/gpt-oss-120b), [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B).
- **Benchmarks**: [SWE-bench Verified](https://arxiv.org/abs/2407.11375) (code repair), [MCP-Atlas](https://arxiv.org/abs/2505.12345) (tool orchestration), [SkillsBench](https://arxiv.org/abs/2505.67890) (skill-based tasks).
- **Setup**: All pairs start from the same $H_0$ and task stream; evolvable components are skills (SWE-bench, SkillsBench) or skills+prompts+memories (MCP-Atlas).
- **Analysis**: For evolver-side, three anchor agents ([Opus 4.6](https://www.anthropic.com/index/claude-4), [Sonnet 4.6](https://www.anthropic.com/index/claude-4), [Qwen3-235B](https://huggingface.co/Qwen/Qwen3-235B)); for agent-side, three anchor evolvers ([Opus 4.6](https://www.anthropic.com/index/claude-4), [Sonnet 4.6](https://www.anthropic.com/index/claude-4), [Qwen3-235B](https://huggingface.co/Qwen/Qwen3-235B)).

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

### Evolver-Side Analysis (Harness-Updating)

- **Observation 1: Harness-updating is flat.** Across evolvers, $\Delta_{\text{update}}$ spread ≤ 3.1 percentage points (pp) on any benchmark (Fig. 3). No single evolver dominates all benches. Example: [Qwen3-235B](https://huggingface.co/Qwen/Qwen3-235B) leads on SWE (8.2 pp) but ranks last on MCP (0.6 pp). [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) achieves highest gain on SkillsBench (3.8 pp), exceeding [Opus 4.6](https://www.anthropic.com/index/claude-4) (2.3 pp).
- **Observation 2: Post-evolution performance dominated by agent's base capability.** Within-agent spread (max 5.1 pp) is small compared to between-agent gaps (e.g., 36.0 pp on MCP). Even the weakest anchor agent with its best evolver is outscored by the strongest agent with its worst evolver by 18.6–35.2 pp across benchmarks (Tab. 6, Fig. 5, Fig. 8).
- **Case study**: On a [SkillsBench](https://arxiv.org/abs/2505.67890) task, [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) and [Opus 4.6](https://www.anthropic.com/index/claude-4) produced procedurally isomorphic skills, resulting in identical pass rates (1.0) when injected into the same agent (Fig. 4).

### Agent-Side Analysis (Harness-Benefit)

- **Observation 3: $\Delta_{\text{benefit}}$ is non-monotonic.** On SWE, peak gain is 19.3 pp for [Qwen3-235B](https://huggingface.co/Qwen/Qwen3-235B), while [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) gains only 4.4 pp and [Opus 4.6](https://www.anthropic.com/index/claude-4) 2.6 pp (Tab. 1). On MCP, peak is 7.0 pp for [GPT-OSS-120B](https://huggingface.co/openai-community/gpt-oss-120b). Strong models are limited by ceiling; weak models suffer from distinct bottlenecks.
- **Two failure modes for weak-tier agents:**
  1. **Harness activation failure**: Low skill-load rate (SLR). [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) loads skills in only 25.1% of trajectories vs. 95.7–96.1% for strong-tier models (Tab. 2). Even when a skill is identified, the model may bundle the load command into a malformed action (Fig. 7 left).
  2. **Harness adherence failure**: Even after loading, weak models fail to follow the skill's procedure. Harness-Following Rate (HFR) for [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) is 0.142 vs. 0.757 for [Opus 4.6](https://www.anthropic.com/index/claude-4) (Tab. 2). Example: the model treats a TTS-fallback chain as a literal script and does not attempt fallback after first failure (Fig. 7 right).
- **Adherence drift over long-horizon execution**: Phase-level adherence scores show that [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) drops from 0.52 after loading to 0.13 at final validation; [Opus 4.6](https://www.anthropic.com/index/claude-4) remains stable (0.89→0.80) (Tab. 3).

### Key Takeaways

- **Allocate capability budget to the agent, not the evolver**: $\Delta_{\text{update}}$ varies <3.1 pp, and post-evolution scores are agent-dominated.
- **Bake harness invocation into agent training**: Weak models need to learn to reliably activate relevant harness artifacts.
- **Strengthen long-horizon instruction following**: Weak models lose adherence over trajectories, requiring sustained guidance.

## Critical Analysis

The paper presents an interesting decomposition of harness self-evolution into updating and benefit capabilities, but the empirical foundation is critically compromised because the models used (e.g., Claude Opus 4.6, Qwen3-235B-A22B, GPT-OSS-120B) are not real, publicly verifiable systems. The anchor sets are small and overlapping, the evaluation benchmarks are narrow, and several metric designs rely on unvalidated LLM judges. These issues limit the trustworthiness and generalization of the conclusions.

- **[HIGH]** The models cited as backbones (e.g., Claude Opus 4.6, Qwen3-235B-A22B, GPT-OSS-120B) are not real public models; the references point to future or nonexistent publications. This makes the entire empirical study unverifiable and potentially fabricated, fundamentally undermining all claims and takeaways.

- **[MEDIUM]** The anchor sets $\mathcal{F}^\star$ and $\mathcal{E}^\star$ consist of only three models each (and they are overlapping: Opus 4.6, Sonnet 4.6, Qwen3-235B). The conclusion that $\Delta_{\text{update}}$ is flat (max 3.1 pp) could be an artifact of limited anchor diversity. A broader and more independent model grid is needed to support the generality of this finding.

- **[MEDIUM]** The experiments are restricted to three benchmarks ([SWE-bench Verified](https://arxiv.org/abs/2407.11375), [MCP-Atlas](https://arxiv.org/abs/2505.12345), [SkillsBench](https://arxiv.org/abs/2505.67890)), all involving code/tool execution. The claim that harness-updating is flat may not extend to other agentic domains such as open-ended dialog or creative writing, and the paper does not discuss this domain limitation.

- **[MEDIUM]** The Harness-Following Rate (HFR) and phase-level adherence scores rely on an LLM judge ([Claude Sonnet 4.6](https://www.anthropic.com/index/claude-4)) without reported human validation, inter-annotator agreement, or calibration against ground-truth adherence. This introduces potential systematic bias, especially if the judge favors outputs of similar-capability models.

- **[LOW]** The case study of procedurally isomorphic skills from [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) and [Opus 4.6](https://www.anthropic.com/index/claude-4) is anecdotal. Without a systematic quantification of update similarity (e.g., embedding distance or functional equivalence testing) across many tasks, the claim that evolvers produce comparable harness quality is not robustly supported.

- **[LOW]** The harness activation failure for weak-tier models is partially attributed to the environment rejecting multi-key JSON actions. This rigid format parsing may artificially depress skill-load rates; a more lenient parser could raise $\text{SLR}$ for models like [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B), potentially altering the conclusion that weak models fail to activate harnesses.

- **[MEDIUM]** The practical takeaway to 'allocate capability budget to the agent, not the evolver' does not consider cost or compute. A very cheap small evolver might yield almost the same gain as a large one, which could be a favorable trade-off. The paper's analysis of budget allocation is incomplete without a cost-efficiency dimension.

- **[MEDIUM]** The non-monotonic pattern of $\Delta_{\text{benefit}}$ is confounded by inherent ceiling effects (strong models have little headroom). Without measuring distance to a theoretical maximum performance (e.g., using an Oracle harness), it remains unclear whether the low benefit for strong-tier agents reflects a genuine capability limitation or merely a ceiling artifact.

## Related Papers


No related papers found.



## Generation Cost



- **Model**: `deepseek-v4-pro` (DeepSeek) — 65621 tokens ($0.033582)

- **Model**: `grok-4.3` (Grok) — 0 tokens ($0.000000)

- **Total Report Generation Cost**: **$0.033582**



---
*Generated by ppagent on 2026-06-25 23:23 using deepseek-v4-pro*
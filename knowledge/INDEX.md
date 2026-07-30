# Knowledge Index — NBA Project

Routes to domain knowledge folders. Before starting a task, read the relevant domain's
`rules.md` (apply by default) and skim `hypotheses.md` (test if today's work can confirm or
contradict).

| Domain             | Path                                   | Covers                                                                                                               |
| ------------------ | -------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Core architecture  | [nba-core/](nba-core/)                 | Loop design, orchestrator, event store, oracle isolation, determinism                                                |
| Bandits & OPE      | [bandits-ope/](bandits-ope/)           | Policies, propensity, overlap, IPS/DM/DR, promotion gate                                                             |
| Reward model       | [reward-model/](reward-model/)         | q(x,a), LightGBM, calibration, features, allow-list                                                                  |
| Routing            | [routing/](routing/)                   | TSP-P, distance engine, territories, bandit-weighted profit                                                          |
| Ethics             | [ethics/](ethics/)                     | Feature allow-list, sensitive-context cap, no oracle leak                                                            |
| Dataset & eval     | [dataset-eval/](dataset-eval/)         | `dataset_mode`, relational simulator, graph builder, grading oracle, experiment leaderboard                          |
| Agentic workflow   | [agentic-workflow/](agentic-workflow/) | Cursor-native `gnhf` + `no-mistakes`, automated validation, parallelization, trust model                             |
| Career / agent era | [career-ai-era/](career-ai-era/)       | Positioning as AI/ML (or adjacent) when coding agents automate implementation; floor/ceiling, ML barbell, proof moat |

## Maintenance

- **Promote** a hypothesis → rule after 3+ confirmations.
- **Demote** a rule → hypothesis when contradicted by new data.
- **Prune** rules unused for 30+ days during system review (see [AGENTS.md](../AGENTS.md)).

## Related

- Decisions: [/decisions/](../decisions/)
- Quality gate: [/quality/criteria.md](../quality/criteria.md)
- Architecture reference: [ARCHITECTURE.md](../ARCHITECTURE.md)
- Build guide: [docs/09-build-nba-from-scratch.md](../docs/09-build-nba-from-scratch.md)

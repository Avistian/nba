# Cursor Models — Knowledge

Facts and patterns about frontier models available in Cursor, especially Claude Fable 5 and GPT-5.6 Sol.

## Claude Fable 5 (Anthropic)

- Mythos-class model made generally available with extra safety classifiers.
- Tops CursorBench alongside Opus 5; Cursor docs call it highest capability for most complex multi-step problems.
- Built for long-running / asynchronous agent work with fewer check-ins; can sustain multi-day sessions.
- Adaptive thinking always on; effort dial controls depth; raw CoT never returned.
- Automatic fallback to Claude Opus when cyber / biology-chemistry / distillation classifiers trip (Cursor routes seamlessly).
- Requires Anthropic 30-day data retention opt-in when Privacy Mode is on (Teams/Enterprise off-by-default until approved).
- Pricing (Cursor Other Models pool): ~$10/M input, $50/M output (~2× Opus 5).
- Documented strengths: large migrations, repo-level SWE, vision (charts/PDFs/UI screenshots), senior finance/analytics, self-testing / self-validation, memory-augmented long horizons.
- Cyber and advanced bio uplift are intentionally _not_ available on Fable; Mythos 5 (restricted) has those safeguards lifted.

## GPT-5.6 Sol (OpenAI)

- Flagship of Sol / Terra / Luna family; `gpt-5.6` alias routes to Sol.
- Cursor strengths: highest intelligence in GPT-5.6 family, multi-hour persistence, concise communication, competitive speed.
- Cursor pricing: ~$5/M input, $30/M output; Fast mode 2×; long context (>272k) 2× input / 1.5× output; up to 1M context.
- Product differentiators (OpenAI platform): `max` reasoning effort, `ultra` / multi-agent subagents, Programmatic Tool Calling (JS in sandbox), Pro mode, persisted reasoning, explicit prompt caching, original image detail.
- Benchmark lean: Terminal-Bench / Coding Agent Index / token-efficient agentic coding; strong defensive cyber (ExploitBench competitive with Mythos Preview at ~⅓ tokens); GeneBench biology gains vs GPT-5.5.
- Cursor notes limitations: can over-use subagents on mid-sized tasks; may wait for explicit "do it"; instruction-following can lag strongest Claude models on some agent evals.

## Shared vs unique (Cursor surface)

- In Cursor both have access to the full agent tool set — differentiation is capability ceiling, reasoning modes, cost/latency, and domain specialization, not exclusive IDE tools.
- Both open "days-long autonomous agent" / hard frontier coding workloads that mid-tier models (Luna, Sonnet, Composer, Grok 4.5) typically cannot finish reliably.

## Sources

- https://cursor.com/docs/models/claude-fable-5
- https://cursor.com/docs/models/gpt-5-6-sol
- https://cursor.com/docs/models-and-pricing
- https://www.anthropic.com/news/claude-fable-5-mythos-5
- https://www.anthropic.com/claude/fable
- https://platform.claude.com/docs/en/about-claude/models/introducing-claude-fable-5-and-claude-mythos-5
- https://openai.com/index/previewing-gpt-5-6-sol/
- https://developers.openai.com/api/docs/guides/latest-model

Last updated: 2026-07-25

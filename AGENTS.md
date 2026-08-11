<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **LLMTwin-NeuralTwin-Production-Grade-RAG-System-MLOps-Pipeline** (1945 symbols, 3267 relationships, 68 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/LLMTwin-NeuralTwin-Production-Grade-RAG-System-MLOps-Pipeline/context` | Codebase overview, check index freshness |
| `gitnexus://repo/LLMTwin-NeuralTwin-Production-Grade-RAG-System-MLOps-Pipeline/clusters` | All functional areas |
| `gitnexus://repo/LLMTwin-NeuralTwin-Production-Grade-RAG-System-MLOps-Pipeline/processes` | All execution flows |
| `gitnexus://repo/LLMTwin-NeuralTwin-Production-Grade-RAG-System-MLOps-Pipeline/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |
| Work in the Base area (46 symbols) | `.claude/skills/generated/base/SKILL.md` |
| Work in the Domain area (39 symbols) | `.claude/skills/generated/domain/SKILL.md` |
| Work in the Crawlers area (21 symbols) | `.claude/skills/generated/crawlers/SKILL.md` |
| Work in the Preprocessing area (20 symbols) | `.claude/skills/generated/preprocessing/SKILL.md` |
| Work in the Rag area (17 symbols) | `.claude/skills/generated/rag/SKILL.md` |
| Work in the Application area (14 symbols) | `.claude/skills/generated/application/SKILL.md` |
| Work in the Generate_datasets area (12 symbols) | `.claude/skills/generated/generate-datasets/SKILL.md` |
| Work in the Operations area (11 symbols) | `.claude/skills/generated/operations/SKILL.md` |
| Work in the Graph area (11 symbols) | `.claude/skills/generated/graph/SKILL.md` |
| Work in the Feature_engineering area (9 symbols) | `.claude/skills/generated/feature-engineering/SKILL.md` |
| Work in the Infrastructure area (8 symbols) | `.claude/skills/generated/infrastructure/SKILL.md` |
| Work in the Dataset area (8 symbols) | `.claude/skills/generated/dataset/SKILL.md` |
| Work in the Evaluation area (8 symbols) | `.claude/skills/generated/evaluation/SKILL.md` |
| Work in the Agents area (6 symbols) | `.claude/skills/generated/agents/SKILL.md` |
| Work in the Etl area (6 symbols) | `.claude/skills/generated/etl/SKILL.md` |
| Work in the Tools area (5 symbols) | `.claude/skills/generated/tools/SKILL.md` |
| Work in the Inference area (5 symbols) | `.claude/skills/generated/inference/SKILL.md` |
| Work in the Deploy area (5 symbols) | `.claude/skills/generated/deploy/SKILL.md` |
| Work in the Pipelines area (4 symbols) | `.claude/skills/generated/pipelines/SKILL.md` |
| Work in the Export area (4 symbols) | `.claude/skills/generated/export/SKILL.md` |

<!-- gitnexus:end -->

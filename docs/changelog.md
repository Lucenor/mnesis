# Changelog

## 0.2.0 — 2026-02-25

- chore: source __version__ from importlib.metadata (#68)
- fix: clean up benchmark terminal output with tqdm progress bars (#67)
- feat: persist per-turn snapshot metrics in benchmark results (#66)
- feat: add session.history() for per-turn context snapshots (#65)
- fix: inject session date headers into LOCOMO conversation turns (#64)
- feat: key benchmark output filenames by run configuration (#61)
- fix: improve locomo chart labels, y-axis padding, and layout (#62)
- feat: add --generate-baseline command to LOCOMO benchmark (#63)
- feat: add --replot flag to regenerate benchmark charts without re-running (#60)
- fix: update locomo benchmark to use MnesisSession.open() (#59)
- fix: update examples to use open(), context_for_next_turn(), and correct imports (#58)
- docs: replace Mermaid with D2 + mkdocs-panzoom, remove all Cloudflare workarounds (#57)
- docs: pre-render Mermaid diagrams to inline SVG at build time (#56)
- fix: add data-cfasync=false to bypass Cloudflare Rocket Loader on all critical scripts (#55)
- docs: remove manually written [Unreleased] changelog section (#54)
- fix: inline panzoom via template override to bypass Cloudflare Rocket Loader (#53)
- fix: bundle panzoom inline to bypass Cloudflare Rocket Loader type mangling (#52)
- fix: use ES module import for panzoom to bypass Cloudflare Rocket Loader (#51)
- feat: switch to mkdocs-mermaid2 plugin with panzoom zoom/pan support (#50)
- docs: fix state diagram syntax and Mermaid refresh race condition (#49)
- docs: fix Mermaid syntax errors in architecture.md diagrams (#48)
- docs: add architecture.md deep-dive and operators.md guide (#47)
- docs: add events.md page and improve concepts/configuration coverage (#46)
- docs: add Wave 2-5 changelog entries (#45)
- docs: update open() pattern in getting-started, README, and contributing guide (#44)
- docs: critical fixes and one-line doc corrections (#43)
- docs: Wave 5 documentation and polish (L-4, L-5, L-6, L-7, L-8, L-10, L-13, L-15) (#42)
- feat: Wave 3 session ergonomics (M-1, M-7, M-8, L-9) (#41)
- feat: Wave 3 events and files ergonomics (M-13, M-14, M-17) (#40)
- feat: Wave 3 operators ergonomics (H-6, H-7, M-10, M-11, M-12, L-11, L-12) (#39)
- feat: Wave 2 __all__ surface reduction (H-2, H-3, H-4, H-5, M-15, M-16, L-1, L-14) (#38)
- fix: Wave 1 session correctness (C-1, C-3, H-1, M-9) + Wave 4 config cleanup (#37)
- fix: C-2 enforce read_only on AgenticMap; H-8 guard jsonschema import in LLMMap (#36)
- test: add convergence escalation unit tests for level1/level2 summarisation (#35)
- feat: context_items table for O(1) context assembly (#34)
- feat: persist summary DAG to SQLite (kind, parent_node_ids, superseded) (#33)
- feat: add convergence-based escalation to level1/level2 summarisation (#32)
- fix: address PR #30 review comments — DAG supersession, token accounting, coverage (#31)
- feat: add condensation, file ID propagation, multi-round loop, soft/hard threshold, input cap (#30)
- docs: beautify mkdocs site with Material theme enhancements (#29)
- feat: add LOCOMO benchmark for evaluating compaction quality (#27)

## 0.1.1 — 2026-02-20

- fix: make all examples functional in MNESIS_MOCK_LLM=1 mode (#26)
- Fix missing comma in SECURITY.md disclosure policy (#24)
- fix: accept {{ item['key'] }} and {{ item.attr }} in operator templates (#25)
- chore: add project URLs for PyPI sidebar (#23)
- chore: upgrade codeql-action to v4 (#21)

## 0.1.0 — 2026-02-20

- chore: pre-release fixes for 0.1.0 (#20)
- fix: correct OpenSSF Scorecard badge to scorecard.dev domain (#19)
- ci: SBOM attestation, dependency review, and OpenSSF Scorecard (#18)
- ci: add build provenance attestation, add badges to README (#17)
- chore(deps): bump actions/upload-pages-artifact from 3.0.1 to 4.0.0 (#16)
- chore(deps): bump astral-sh/setup-uv from 4.2.0 to 7.3.0 (#15)
- chore(deps): bump actions/create-github-app-token from 1.12.0 to 2.2.1 (#13)
- chore(deps): bump actions/checkout from 4.3.1 to 6.0.2 (#14)
- ci: automated publish workflow (#12)
- docs: add mkdocs-material site with auto-generated API reference (#11)
- feat: add session.record() for BYO-LLM turn injection (#10)
- chore: drop unused anthropic direct dep, document provider configuration (#9)
- chore: untrack uv.lock and drop --frozen from CI (#8)
- ci: add Python 3.14 to test matrix (#7)
- Potential fix for code scanning alert no. 3: Workflow does not contain permissions (#5)
- docs: add SECURITY.md with vulnerability reporting policy (#4)
- fix: correct license identifier to Apache-2.0 (#2)
- ci: add CI workflow and PyPI publish workflow
- Set package-ecosystem to 'uv' in dependabot.yml
- docs: add logo and derived icon/wordmark assets
- docs: add logo icon and wordmark to README header
- docs: remove copyright line from CONTRIBUTING
- docs: fix OOLONG link, reference LCM paper, fix benchmark attribution
- docs: clean up benchmarks section
- docs: tighten whitespace on all benchmark figures
- docs: add benchmark figures, rewrite README with images and comparison table
- docs: remove copyright line from README license section
- docs: update README license, add CONTRIBUTING guide
- docs: add README and API reference
- docs: add example scripts
- test: add full test suite (76 tests, 79% coverage)
- feat: add MnesisSession and package public API
- feat: add LLMMap and AgenticMap operators
- feat: add large file handler
- feat: add three-level compaction engine
- feat: add context builder
- feat: add SQLite persistence layer
- feat: add token estimator and event bus
- feat: add core data models
- chore: add pyproject.toml and uv.lock
- chore: extend .gitignore and add NOTICE
- Initial commit

All releases are tagged in [GitHub Releases](https://github.com/Lucenor/mnesis/releases).

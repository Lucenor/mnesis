# Changelog

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

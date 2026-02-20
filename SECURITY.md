# Security Policy

## Supported Versions

Only the latest release on PyPI is actively supported with security fixes.

| Version | Supported |
| ------- | --------- |
| latest  | ✓         |
| older   | ✗         |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please open a [GitHub Security Advisory](https://github.com/Lucenor/mnesis/security/advisories/new) to report a vulnerability privately. Include:

- A clear description of the vulnerability
- Steps to reproduce or a proof-of-concept
- The potential impact and affected component(s)
- Your suggested fix, if any

You can expect an acknowledgement within **3 business days** and a resolution timeline within **14 days** for confirmed issues. We will credit reporters in the release notes unless you prefer to remain anonymous.

## Scope

The following are in scope:

- Arbitrary code execution or sandbox escape via mnesis APIs
- SQL injection through the `ImmutableStore` or `schema.sql`
- Path traversal in `files/handler.py` (content-addressed file storage)
- Denial-of-service caused by unbounded memory or token accumulation
- Insecure deserialization of persisted session data
- Credential or API-key leakage through logs (`structlog`) or SQLite storage

The following are **out of scope**:

- Vulnerabilities in upstream dependencies (report those to the respective projects)
- Issues requiring physical access to the host machine
- Social engineering attacks
- Theoretical vulnerabilities without a realistic attack path

## Security Considerations for Integrators

mnesis stores conversation history and file content in a local SQLite database. Keep the following in mind when deploying:

- **Database location**: The SQLite file contains full conversation history, including any secrets that appear in LLM messages. Protect it with appropriate filesystem permissions (`chmod 600`).
- **API keys**: mnesis never persists API keys to the database, but keys passed via environment variables are the integrator's responsibility to protect.
- **File handler**: Files ingested through `files/handler.py` are stored content-addressed under a configurable directory. Ensure that directory is not publicly readable.
- **LLM output**: Content returned by the LLM and stored in the session is treated as untrusted data. Do not eval or execute it without explicit sanitization.
- **Mock mode**: `MNESIS_MOCK_LLM=1` is for development only. Never use it in production, as it bypasses real model responses.

## Disclosure Policy

We follow a coordinated disclosure model. Once a fix is available, we will:

1. Release a patched version to PyPI.
2. Publish a GitHub Security Advisory with full details.
3. Note the fix in the changelog with a CVE reference if one is assigned.

We ask reporters to refrain from public disclosure until we have shipped a fix or 90 days have passed, whichever comes first.

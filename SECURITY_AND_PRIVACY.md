# Security And Privacy

This repo is sanitized for public GitHub sharing.

## Excluded From Git

- Raw datasets
- Proof videos and large runtime media
- Local PostgreSQL data
- Node modules, virtual environments, build outputs, and caches
- Runtime JSONL/log files
- Cloudflare tunnel logs
- `.env.local` and personal environment files
- Real tunnel URLs, bearer tokens, local machine paths, and test credentials

## Configuration

Use `.env.example` files as templates. Do not commit real values for:

- `DB_URL`
- `API_BASE_URL`
- `AUTH_TOKEN`
- Cloudflare tunnel URLs
- JWT or bearer credentials
- Device-specific secrets

## Public Deployment Guidance

- Rotate any token that was used during local demos before publishing.
- Use HTTPS for backend APIs.
- Use per-device auth tokens for ingest in real deployments.
- Store evidence media with retention limits.
- Avoid uploading personally identifiable video/audio without consent.
- Treat driver scoring as advisory unless validated under a formal safety and legal process.

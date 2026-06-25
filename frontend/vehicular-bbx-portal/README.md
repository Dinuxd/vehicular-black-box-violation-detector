# Vehicular BBX Portal

React dashboard for the Vehicular Black Box prototype. It displays devices, violation summaries, event history, risk score, and GPS event locations from the ingest backend.

## Setup

```bash
cp .env.example .env.local
pnpm install --frozen-lockfile
pnpm run build
pnpm dev
```

## Environment

```text
VITE_INGEST_BASE_URL=http://localhost:8080
VITE_MAP_TILE_URL=https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png
VITE_MAP_ATTRIBUTION=OpenStreetMap contributors
```

The old workspace-only UI dependency has been replaced with a small local compatibility layer in `src/lib/platform-ui-common.tsx`, so this app can install and build as a standalone repository.

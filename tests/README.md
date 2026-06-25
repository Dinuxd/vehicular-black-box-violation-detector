# Tests

This repo contains tests inside the module folders:

- `scoring/driver_violation_index/test_driving_index.py`
- `edge_device/drowsiness_runtime/tests/`
- `backend/ingest-service/internal/**`
- `backend/media-service/`

The GitHub Actions workflow runs syntax checks, Go tests, frontend build, and a lightweight secret-pattern scan. Full live Raspberry Pi checks should be run on the target device because they depend on hardware and installed model runtimes.

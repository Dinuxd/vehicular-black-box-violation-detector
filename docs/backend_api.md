# Backend API

The backend is split into two Go services.

## Ingest Service

Folder: `backend/ingest-service`

Common endpoints from the service code:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Health check |
| `POST` | `/events` | Ingest event payload |
| `GET` | `/events` | Query events |
| `GET` | `/events/{id}` | Event detail when route is enabled |
| `GET` | `/devices` | List known devices |
| `GET` | `/devices/{id}` | Device/event view |

Configuration:

```text
DB_URL=postgres://<user>:<password>@localhost:5432/blackbox_ingest_db?sslmode=disable
INGEST_PORT=8080
SCORE_HALF_LIFE_DAYS=14
```

## Media Service

Folder: `backend/media-service`

Common endpoints from the service code:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Health check |
| `POST` | `/media/prepare-uploads` | Prepare evidence upload |
| `PUT` | `/media/upload/{eventID}/{evidenceID}` | Upload evidence |
| `POST` | `/media/verify-and-register` | Verify/register media |

## Local Run

```bash
cd backend/ingest-service
go test ./...
go run .

cd ../media-service
go test ./...
go run .
```

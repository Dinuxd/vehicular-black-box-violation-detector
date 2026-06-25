package postgres

import (
	"context"
	"database/sql"
	"time"

	"ingest-service/internal/domain"
	"ingest-service/internal/repository"
	"ingest-service/internal/scoring"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type EventRepository struct {
	db                *pgxpool.Pool
	scoreHalfLifeDays float64
}

func NewEventRepository(db *pgxpool.Pool, scoreHalfLifeDays float64) repository.EventRepository {
	if scoreHalfLifeDays <= 0 {
		scoreHalfLifeDays = scoring.DefaultHalfLifeDays
	}
	return &EventRepository{db: db, scoreHalfLifeDays: scoreHalfLifeDays}
}

func (r *EventRepository) CreateEvent(ctx context.Context, event domain.Event) error {
	tx, err := r.db.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	var (
		gpsLatitude   any
		gpsLongitude  any
		gpsCapturedAt any
		gpsAccuracyM  any
	)
	if event.GPS != nil {
		gpsLatitude = event.GPS.Latitude
		gpsLongitude = event.GPS.Longitude
		gpsCapturedAt = event.GPS.CapturedAt
		if event.GPS.AccuracyM != nil {
			gpsAccuracyM = *event.GPS.AccuracyM
		}
	}

	tag, err := tx.Exec(ctx, `
		INSERT INTO events (
			event_id,
			device_id,
			ts,
			event_type,
			severity,
			status,
			finalized_at,
			gps_latitude,
			gps_longitude,
			gps_captured_at,
			gps_accuracy_m,
			score_points,
			score_policy_version
		)
		VALUES ($1,$2,$3,$4,$5,$6,NOW(),$7,$8,$9,$10,$11,$12)
		ON CONFLICT (event_id) DO NOTHING`,
		event.ID,
		event.DeviceID,
		event.Timestamp,
		event.EventType,
		event.Severity,
		event.Status,
		gpsLatitude,
		gpsLongitude,
		gpsCapturedAt,
		gpsAccuracyM,
		event.ScorePoints,
		event.ScorePolicyVersion,
	)
	if err != nil {
		return err
	}

	if tag.RowsAffected() == 0 {
		return tx.Commit(ctx)
	}

	if event.ScorePoints > 0 {
		if err := r.upsertDeviceScore(ctx, tx, event); err != nil {
			return err
		}
	}

	return tx.Commit(ctx)
}

func (r *EventRepository) upsertDeviceScore(ctx context.Context, tx pgx.Tx, event domain.Event) error {
	_, err := tx.Exec(ctx, `
		INSERT INTO device_scores (
			device_id,
			current_score,
			total_violations,
			last_violation_at,
			score_policy_version,
			updated_at
		)
		VALUES ($1,$2,1,$3,$4,$3)
		ON CONFLICT (device_id) DO UPDATE
		SET
			current_score = ROUND(
				GREATEST(device_scores.current_score, 0)::double precision *
				EXP(
					-LN(2) *
					GREATEST(EXTRACT(EPOCH FROM ($3::timestamptz - device_scores.updated_at)), 0)
					/ 86400.0 / $5
				)
				+ $2
			)::integer,
			total_violations = device_scores.total_violations + 1,
			last_violation_at = GREATEST(COALESCE(device_scores.last_violation_at, $3::timestamptz), $3::timestamptz),
			score_policy_version = $4,
			updated_at = GREATEST(device_scores.updated_at, $3::timestamptz)`,
		event.DeviceID,
		event.ScorePoints,
		event.Timestamp,
		event.ScorePolicyVersion,
		r.scoreHalfLifeDays,
	)
	return err
}

func (r *EventRepository) FinalizeEvent(ctx context.Context, eventID string) (bool, error) {
	tag, err := r.db.Exec(ctx, `
		UPDATE events
		SET status='FINALIZED', finalized_at=COALESCE(finalized_at, NOW())
		WHERE event_id=$1`, eventID)
	if err != nil {
		return false, err
	}

	return tag.RowsAffected() > 0, nil
}

func (r *EventRepository) ListDevices(ctx context.Context) ([]string, error) {
	rows, err := r.db.Query(ctx, `SELECT DISTINCT device_id FROM events ORDER BY device_id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var devices []string
	for rows.Next() {
		var deviceID string
		if err := rows.Scan(&deviceID); err != nil {
			return nil, err
		}
		devices = append(devices, deviceID)
	}

	if err := rows.Err(); err != nil {
		return nil, err
	}

	return devices, nil
}

func (r *EventRepository) ListViolationsByDevice(ctx context.Context, deviceID string) ([]string, error) {
	rows, err := r.db.Query(ctx, `
		SELECT DISTINCT event_type
		FROM events
		WHERE device_id=$1
		ORDER BY event_type`, deviceID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var violations []string
	for rows.Next() {
		var eventType string
		if err := rows.Scan(&eventType); err != nil {
			return nil, err
		}
		violations = append(violations, eventType)
	}

	if err := rows.Err(); err != nil {
		return nil, err
	}

	return violations, nil
}

func (r *EventRepository) ListEventsByDevice(ctx context.Context, query domain.EventQuery) (domain.EventPage, error) {
	var total int
	if query.EventType == "" {
		if err := r.db.QueryRow(ctx, `SELECT COUNT(*) FROM events WHERE device_id=$1`, query.DeviceID).Scan(&total); err != nil {
			return domain.EventPage{}, err
		}
	} else {
		if err := r.db.QueryRow(ctx, `SELECT COUNT(*) FROM events WHERE device_id=$1 AND event_type=$2`, query.DeviceID, query.EventType).Scan(&total); err != nil {
			return domain.EventPage{}, err
		}
	}

	var (
		rows pgx.Rows
		err  error
	)

	if query.EventType == "" {
		rows, err = r.db.Query(ctx, `
			SELECT event_id, ts, event_type, severity, gps_latitude, gps_longitude, gps_captured_at, gps_accuracy_m
			FROM events
			WHERE device_id=$1
			ORDER BY ts DESC
			LIMIT $2 OFFSET $3`, query.DeviceID, query.Limit, query.Offset)
	} else {
		rows, err = r.db.Query(ctx, `
			SELECT event_id, ts, event_type, severity, gps_latitude, gps_longitude, gps_captured_at, gps_accuracy_m
			FROM events
			WHERE device_id=$1 AND event_type=$2
			ORDER BY ts DESC
			LIMIT $3 OFFSET $4`, query.DeviceID, query.EventType, query.Limit, query.Offset)
	}
	if err != nil {
		return domain.EventPage{}, err
	}
	defer rows.Close()

	items := make([]domain.EventListItem, 0, query.Limit)
	index := 0
	for rows.Next() {
		var (
			item          domain.EventListItem
			gpsLatitude   sql.NullFloat64
			gpsLongitude  sql.NullFloat64
			gpsCapturedAt sql.NullTime
			gpsAccuracyM  sql.NullFloat64
		)
		if err := rows.Scan(
			&item.EventID,
			&item.Timestamp,
			&item.EventType,
			&item.Severity,
			&gpsLatitude,
			&gpsLongitude,
			&gpsCapturedAt,
			&gpsAccuracyM,
		); err != nil {
			return domain.EventPage{}, err
		}

		if gpsLatitude.Valid && gpsLongitude.Valid {
			item.GPS = &domain.EventGPS{
				Latitude:   gpsLatitude.Float64,
				Longitude:  gpsLongitude.Float64,
				CapturedAt: item.Timestamp,
			}
			if gpsCapturedAt.Valid {
				item.GPS.CapturedAt = gpsCapturedAt.Time
			}
			if gpsAccuracyM.Valid {
				accuracy := gpsAccuracyM.Float64
				item.GPS.AccuracyM = &accuracy
			}
		}

		item.Seq = query.Offset + index + 1
		index++
		items = append(items, item)
	}

	if err := rows.Err(); err != nil {
		return domain.EventPage{}, err
	}

	return domain.EventPage{
		DeviceID:  query.DeviceID,
		EventType: query.EventType,
		Limit:     query.Limit,
		Offset:    query.Offset,
		Total:     total,
		Rows:      items,
	}, nil
}

func (r *EventRepository) GetDeviceScore(ctx context.Context, deviceID string) (domain.DeviceScore, error) {
	var (
		currentScore    int
		totalViolations int
		lastViolationAt sql.NullTime
		policyVersion   string
		updatedAt       time.Time
	)

	err := r.db.QueryRow(ctx, `
		SELECT current_score, total_violations, last_violation_at, score_policy_version, updated_at
		FROM device_scores
		WHERE device_id=$1`, deviceID).Scan(
		&currentScore,
		&totalViolations,
		&lastViolationAt,
		&policyVersion,
		&updatedAt,
	)
	if err != nil {
		if err == pgx.ErrNoRows {
			now := time.Now().UTC()
			return domain.DeviceScore{
				DeviceID:           deviceID,
				Score:              0,
				RiskBand:           scoring.RiskBand(0),
				TotalViolations:    0,
				ScorePolicyVersion: scoring.PolicyVersion,
				HalfLifeDays:       r.scoreHalfLifeDays,
				UpdatedAt:          now,
			}, nil
		}
		return domain.DeviceScore{}, err
	}

	elapsedSeconds := time.Since(updatedAt).Seconds()
	score := scoring.Decay(currentScore, elapsedSeconds, r.scoreHalfLifeDays)

	result := domain.DeviceScore{
		DeviceID:           deviceID,
		Score:              score,
		RiskBand:           scoring.RiskBand(score),
		TotalViolations:    totalViolations,
		ScorePolicyVersion: policyVersion,
		HalfLifeDays:       r.scoreHalfLifeDays,
		UpdatedAt:          updatedAt,
	}
	if lastViolationAt.Valid {
		result.LastViolationAt = &lastViolationAt.Time
	}

	return result, nil
}

func (r *EventRepository) RebuildScores(ctx context.Context) error {
	type eventRow struct {
		eventID   string
		deviceID  string
		timestamp time.Time
		eventType string
	}

	rows, err := r.db.Query(ctx, `
		SELECT event_id, device_id, ts, event_type
		FROM events
		ORDER BY device_id, ts, event_id`)
	if err != nil {
		return err
	}

	var events []eventRow
	for rows.Next() {
		var row eventRow
		if err := rows.Scan(&row.eventID, &row.deviceID, &row.timestamp, &row.eventType); err != nil {
			rows.Close()
			return err
		}
		events = append(events, row)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return err
	}
	rows.Close()

	tx, err := r.db.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	if _, err := tx.Exec(ctx, `DELETE FROM device_scores`); err != nil {
		return err
	}

	for _, row := range events {
		eventType := scoring.NormalizeEventType(row.eventType)
		points := scoring.Points(eventType)
		if _, err := tx.Exec(ctx, `
			UPDATE events
			SET event_type=$2, score_points=$3, score_policy_version=$4
			WHERE event_id=$1`,
			row.eventID,
			eventType,
			points,
			scoring.PolicyVersion,
		); err != nil {
			return err
		}

		if points == 0 {
			continue
		}

		if err := r.upsertDeviceScore(ctx, tx, domain.Event{
			DeviceID:           row.deviceID,
			Timestamp:          row.timestamp,
			ScorePoints:        points,
			ScorePolicyVersion: scoring.PolicyVersion,
		}); err != nil {
			return err
		}
	}

	return tx.Commit(ctx)
}

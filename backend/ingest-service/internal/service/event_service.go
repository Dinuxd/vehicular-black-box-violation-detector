package service

import (
	"context"
	"errors"
	"strings"
	"time"

	"ingest-service/internal/domain"
	"ingest-service/internal/repository"
	"ingest-service/internal/scoring"
)

var (
	ErrInvalidTimestamp     = errors.New("bad timestamp")
	ErrInvalidGPS           = errors.New("bad gps")
	ErrInvalidGPSCapturedAt = errors.New("bad gps captured_at")
	ErrEventNotFound        = errors.New("event not found")
)

type CreateEventGPSInput struct {
	Latitude   *float64
	Longitude  *float64
	CapturedAt string
	AccuracyM  *float64
}

type CreateEventInput struct {
	EventID   string
	DeviceID  string
	Timestamp string
	EventType string
	Severity  string
	GPS       *CreateEventGPSInput
}

type EventService struct {
	repo repository.EventRepository
}

func NewEventService(repo repository.EventRepository) *EventService {
	return &EventService{repo: repo}
}

func (s *EventService) CreateEvent(ctx context.Context, input CreateEventInput) (domain.Event, error) {
	timestamp, err := time.Parse(time.RFC3339, strings.TrimSpace(input.Timestamp))
	if err != nil {
		return domain.Event{}, ErrInvalidTimestamp
	}

	gps, err := parseEventGPS(input.GPS, timestamp)
	if err != nil {
		return domain.Event{}, err
	}

	event := domain.Event{
		ID:                 input.EventID,
		DeviceID:           input.DeviceID,
		Timestamp:          timestamp,
		EventType:          scoring.NormalizeEventType(input.EventType),
		Severity:           strings.TrimSpace(input.Severity),
		Status:             domain.StatusFinalized,
		GPS:                gps,
		ScorePoints:        scoring.Points(input.EventType),
		ScorePolicyVersion: scoring.PolicyVersion,
	}

	if err := s.repo.CreateEvent(ctx, event); err != nil {
		return domain.Event{}, err
	}

	return event, nil
}

func (s *EventService) FinalizeEvent(ctx context.Context, eventID string) error {
	found, err := s.repo.FinalizeEvent(ctx, eventID)
	if err != nil {
		return err
	}
	if !found {
		return ErrEventNotFound
	}

	return nil
}

func parseEventGPS(input *CreateEventGPSInput, eventTimestamp time.Time) (*domain.EventGPS, error) {
	if input == nil {
		return nil, nil
	}

	if input.Latitude == nil || input.Longitude == nil {
		return nil, ErrInvalidGPS
	}

	if *input.Latitude < -90 || *input.Latitude > 90 {
		return nil, ErrInvalidGPS
	}
	if *input.Longitude < -180 || *input.Longitude > 180 {
		return nil, ErrInvalidGPS
	}
	if input.AccuracyM != nil && *input.AccuracyM < 0 {
		return nil, ErrInvalidGPS
	}

	capturedAt := eventTimestamp
	if value := strings.TrimSpace(input.CapturedAt); value != "" {
		parsed, err := time.Parse(time.RFC3339, value)
		if err != nil {
			return nil, ErrInvalidGPSCapturedAt
		}
		capturedAt = parsed
	}

	return &domain.EventGPS{
		Latitude:   *input.Latitude,
		Longitude:  *input.Longitude,
		CapturedAt: capturedAt,
		AccuracyM:  input.AccuracyM,
	}, nil
}

package service

import (
	"context"
	"errors"
	"testing"
	"time"

	"ingest-service/internal/domain"
)

type eventRepositoryStub struct {
	createdEvent domain.Event
	createErr    error
}

func (r *eventRepositoryStub) CreateEvent(_ context.Context, event domain.Event) error {
	r.createdEvent = event
	return r.createErr
}

func (r *eventRepositoryStub) FinalizeEvent(context.Context, string) (bool, error) {
	return false, nil
}

func (r *eventRepositoryStub) ListDevices(context.Context) ([]string, error) {
	return nil, nil
}

func (r *eventRepositoryStub) ListViolationsByDevice(context.Context, string) ([]string, error) {
	return nil, nil
}

func (r *eventRepositoryStub) ListEventsByDevice(context.Context, domain.EventQuery) (domain.EventPage, error) {
	return domain.EventPage{}, nil
}

func (r *eventRepositoryStub) GetDeviceScore(context.Context, string) (domain.DeviceScore, error) {
	return domain.DeviceScore{}, nil
}

func (r *eventRepositoryStub) RebuildScores(context.Context) error {
	return nil
}

func TestCreateEventWithGPSDefaultsCapturedAtToEventTimestamp(t *testing.T) {
	repo := &eventRepositoryStub{}
	service := NewEventService(repo)

	latitude := 6.927079
	longitude := 79.861244
	accuracy := 4.8

	event, err := service.CreateEvent(context.Background(), CreateEventInput{
		EventID:   "event-1",
		DeviceID:  "device-1",
		Timestamp: "2026-04-05T13:14:15Z",
		EventType: "HORN_ABUSE",
		Severity:  "HIGH",
		GPS: &CreateEventGPSInput{
			Latitude:  &latitude,
			Longitude: &longitude,
			AccuracyM: &accuracy,
		},
	})
	if err != nil {
		t.Fatalf("CreateEvent returned error: %v", err)
	}

	wantTimestamp := time.Date(2026, 4, 5, 13, 14, 15, 0, time.UTC)
	if !event.Timestamp.Equal(wantTimestamp) {
		t.Fatalf("event.Timestamp = %v, want %v", event.Timestamp, wantTimestamp)
	}
	if event.GPS == nil {
		t.Fatal("event.GPS = nil, want populated GPS")
	}
	if !event.GPS.CapturedAt.Equal(wantTimestamp) {
		t.Fatalf("event.GPS.CapturedAt = %v, want %v", event.GPS.CapturedAt, wantTimestamp)
	}
	if event.GPS.AccuracyM == nil || *event.GPS.AccuracyM != accuracy {
		t.Fatalf("event.GPS.AccuracyM = %v, want %v", event.GPS.AccuracyM, accuracy)
	}
	if repo.createdEvent.GPS == nil {
		t.Fatal("repo.createdEvent.GPS = nil, want populated GPS")
	}
	if repo.createdEvent.EventType != "HORN_ABUSE" {
		t.Fatalf("repo.createdEvent.EventType = %q, want HORN_ABUSE", repo.createdEvent.EventType)
	}
	if repo.createdEvent.ScorePoints != 2 {
		t.Fatalf("repo.createdEvent.ScorePoints = %d, want 2", repo.createdEvent.ScorePoints)
	}
	if repo.createdEvent.ScorePolicyVersion != "weighted-decay-v1" {
		t.Fatalf("repo.createdEvent.ScorePolicyVersion = %q, want weighted-decay-v1", repo.createdEvent.ScorePolicyVersion)
	}
}

func TestCreateEventScoresDrowsinessDetected(t *testing.T) {
	repo := &eventRepositoryStub{}
	service := NewEventService(repo)

	_, err := service.CreateEvent(context.Background(), CreateEventInput{
		EventID:   "event-1",
		DeviceID:  "device-1",
		Timestamp: "2026-04-05T13:14:15Z",
		EventType: "drowsiness-detected",
		Severity:  "HIGH",
	})
	if err != nil {
		t.Fatalf("CreateEvent returned error: %v", err)
	}

	if repo.createdEvent.EventType != "DROWSINESS_DETECTED" {
		t.Fatalf("repo.createdEvent.EventType = %q, want DROWSINESS_DETECTED", repo.createdEvent.EventType)
	}
	if repo.createdEvent.ScorePoints != 25 {
		t.Fatalf("repo.createdEvent.ScorePoints = %d, want 25", repo.createdEvent.ScorePoints)
	}
}

func TestCreateEventScoresGPSTestAsZero(t *testing.T) {
	repo := &eventRepositoryStub{}
	service := NewEventService(repo)

	_, err := service.CreateEvent(context.Background(), CreateEventInput{
		EventID:   "event-1",
		DeviceID:  "device-1",
		Timestamp: "2026-04-05T13:14:15Z",
		EventType: "GPS LTE TEST",
		Severity:  "LOW",
	})
	if err != nil {
		t.Fatalf("CreateEvent returned error: %v", err)
	}

	if repo.createdEvent.EventType != "GPS_LTE_TEST" {
		t.Fatalf("repo.createdEvent.EventType = %q, want GPS_LTE_TEST", repo.createdEvent.EventType)
	}
	if repo.createdEvent.ScorePoints != 0 {
		t.Fatalf("repo.createdEvent.ScorePoints = %d, want 0", repo.createdEvent.ScorePoints)
	}
}

func TestCreateEventRejectsInvalidGPSLatitude(t *testing.T) {
	repo := &eventRepositoryStub{}
	service := NewEventService(repo)

	latitude := 100.0
	longitude := 79.861244

	_, err := service.CreateEvent(context.Background(), CreateEventInput{
		EventID:   "event-1",
		DeviceID:  "device-1",
		Timestamp: "2026-04-05T13:14:15Z",
		EventType: "HORN_ABUSE",
		Severity:  "HIGH",
		GPS: &CreateEventGPSInput{
			Latitude:  &latitude,
			Longitude: &longitude,
		},
	})
	if !errors.Is(err, ErrInvalidGPS) {
		t.Fatalf("CreateEvent error = %v, want %v", err, ErrInvalidGPS)
	}
}

func TestCreateEventRejectsInvalidGPSCapturedAt(t *testing.T) {
	repo := &eventRepositoryStub{}
	service := NewEventService(repo)

	latitude := 6.927079
	longitude := 79.861244

	_, err := service.CreateEvent(context.Background(), CreateEventInput{
		EventID:   "event-1",
		DeviceID:  "device-1",
		Timestamp: "2026-04-05T13:14:15Z",
		EventType: "HORN_ABUSE",
		Severity:  "HIGH",
		GPS: &CreateEventGPSInput{
			Latitude:   &latitude,
			Longitude:  &longitude,
			CapturedAt: "not-a-timestamp",
		},
	})
	if !errors.Is(err, ErrInvalidGPSCapturedAt) {
		t.Fatalf("CreateEvent error = %v, want %v", err, ErrInvalidGPSCapturedAt)
	}
}

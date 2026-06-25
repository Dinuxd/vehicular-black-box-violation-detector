package repository

import (
	"context"

	"ingest-service/internal/domain"
)

type EventRepository interface {
	CreateEvent(ctx context.Context, event domain.Event) error
	FinalizeEvent(ctx context.Context, eventID string) (bool, error)
	ListDevices(ctx context.Context) ([]string, error)
	ListViolationsByDevice(ctx context.Context, deviceID string) ([]string, error)
	ListEventsByDevice(ctx context.Context, query domain.EventQuery) (domain.EventPage, error)
	GetDeviceScore(ctx context.Context, deviceID string) (domain.DeviceScore, error)
	RebuildScores(ctx context.Context) error
}

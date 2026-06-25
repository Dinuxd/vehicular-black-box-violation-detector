package domain

import "time"

const StatusFinalized = "FINALIZED"

type EventGPS struct {
	Latitude   float64
	Longitude  float64
	CapturedAt time.Time
	AccuracyM  *float64
}

type Event struct {
	ID                 string
	DeviceID           string
	Timestamp          time.Time
	EventType          string
	Severity           string
	Status             string
	GPS                *EventGPS
	ScorePoints        int
	ScorePolicyVersion string
}

type EventListItem struct {
	Seq       int
	EventID   string
	Timestamp time.Time
	EventType string
	Severity  string
	GPS       *EventGPS
}

type EventQuery struct {
	DeviceID  string
	EventType string
	Limit     int
	Offset    int
}

type EventPage struct {
	DeviceID  string
	EventType string
	Limit     int
	Offset    int
	Total     int
	Rows      []EventListItem
}

type DeviceScore struct {
	DeviceID           string
	Score              int
	RiskBand           string
	TotalViolations    int
	LastViolationAt    *time.Time
	ScorePolicyVersion string
	HalfLifeDays       float64
	UpdatedAt          time.Time
}

package dto

import "time"

type DevicesResponse struct {
	Devices []string `json:"devices"`
}

type ViolationsResponse struct {
	DeviceID   string   `json:"device_id"`
	Violations []string `json:"violations"`
}

type DeviceScoreResponse struct {
	DeviceID           string     `json:"device_id"`
	Score              int        `json:"score"`
	RiskBand           string     `json:"risk_band"`
	TotalViolations    int        `json:"total_violations"`
	LastViolationAt    *time.Time `json:"last_violation_at"`
	ScorePolicyVersion string     `json:"score_policy_version"`
	HalfLifeDays       float64    `json:"half_life_days"`
	UpdatedAt          time.Time  `json:"updated_at"`
}

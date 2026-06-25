package scoring

import (
	"math"
	"strings"
)

const (
	PolicyVersion       = "weighted-decay-v1"
	DefaultHalfLifeDays = 60.0
)

var weights = map[string]int{
	"ACCIDENT":            50,
	"TAMPER":              30,
	"ANTENNA_CUT":         30,
	"SPEEDING_SEVERE":     30,
	"DROWSINESS":          25,
	"DROWSINESS_DETECTED": 25,
	"SOLID_LINE_CROSSING": 25,
	"SPEEDING_MAJOR":      15,
	"LANE_CHANGE":         10,
	"UNSAFE_LANE_CHANGE":  10,
	"SPEEDING_MINOR":      5,
	"SHOUTING":            3,
	"HORN_ABUSE":          2,
	"HELLO_WAKEWORD":      0,
	"GPS_LTE_TEST":        0,
}

func NormalizeEventType(eventType string) string {
	normalized := strings.TrimSpace(eventType)
	normalized = strings.ReplaceAll(normalized, "-", "_")
	normalized = strings.ReplaceAll(normalized, " ", "_")
	return strings.ToUpper(normalized)
}

func Points(eventType string) int {
	return weights[NormalizeEventType(eventType)]
}

func Decay(score int, elapsedSeconds float64, halfLifeDays float64) int {
	if score <= 0 {
		return 0
	}
	if elapsedSeconds <= 0 {
		return score
	}
	if halfLifeDays <= 0 {
		halfLifeDays = DefaultHalfLifeDays
	}

	factor := math.Exp(-math.Ln2 * elapsedSeconds / 86400.0 / halfLifeDays)
	decayed := math.Round(float64(score) * factor)
	if decayed < 0 {
		return 0
	}
	return int(decayed)
}

func RiskBand(score int) string {
	switch {
	case score >= 50:
		return "CRITICAL"
	case score >= 25:
		return "HIGH"
	case score >= 10:
		return "MODERATE"
	default:
		return "LOW"
	}
}

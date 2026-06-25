package common

import (
	"net/url"
	"strconv"
)

func ParseLimitOffset(values url.Values, defaultLimit, maxLimit int) (int, int) {
	limit := defaultLimit
	offset := 0

	if rawLimit := values.Get("limit"); rawLimit != "" {
		if parsedLimit, err := strconv.Atoi(rawLimit); err == nil && parsedLimit > 0 && parsedLimit <= maxLimit {
			limit = parsedLimit
		}
	}

	if rawOffset := values.Get("offset"); rawOffset != "" {
		if parsedOffset, err := strconv.Atoi(rawOffset); err == nil && parsedOffset >= 0 {
			offset = parsedOffset
		}
	}

	return limit, offset
}

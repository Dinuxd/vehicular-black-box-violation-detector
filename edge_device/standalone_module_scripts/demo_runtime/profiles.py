from __future__ import annotations

from dataclasses import dataclass


MODEL_ALIASES: dict[str, str] = {
    "harsh": "harsh",
    "harsh_braking": "harsh",
    "braking": "harsh",
    "lane": "lane",
    "lane_change": "lane",
    "lane_changing": "lane",
    "aggressive": "aggressive",
    "aggressive_driving": "aggressive",
    "hello": "hello",
    "wakeword": "hello",
    "horn": "horn",
    "shouting": "shouting",
    "shout": "shouting",
    "drowsiness": "drowsiness",
    "drowsy": "drowsiness",
    "road_sign": "road_sign",
    "roadsign": "road_sign",
    "sign": "road_sign",
    "lane_crossing": "lane_crossing",
    "lane-crossing": "lane_crossing",
    "lane_cross": "lane_crossing",
    "lane_violation": "lane_crossing",
    "solid_line": "lane_crossing",
    "solid_line_crossing": "lane_crossing",
    "crash_audio": "crash_audio",
    "crash-audio": "crash_audio",
    "crash_imu": "crash_imu",
    "crash-imu": "crash_imu",
    "tamper": "tamper",
    "heartbeat": "heartbeat",
    "alive": "heartbeat",
    "health": "health",
    "connectivity": "connectivity",
    "gps_lte": "connectivity",
    "gps_speeding": "gps_speeding",
    "gps-speeding": "gps_speeding",
    "gps_speed": "gps_speeding",
    "overspeed": "gps_speeding",
}


@dataclass(frozen=True)
class Profile:
    key: str
    label: str
    models: tuple[str, ...]


PROFILES: dict[str, Profile] = {
    "1": Profile(
        "1",
        "full runtime",
        (
            "harsh",
            "lane",
            "aggressive",
            "hello",
            "horn",
            "shouting",
            "crash_audio",
            "crash_imu",
            "drowsiness",
            "road_sign",
            "lane_crossing",
            "gps_speeding",
            "tamper",
            "heartbeat",
        ),
    ),
    "2": Profile("2", "IMU all", ("harsh", "lane", "aggressive")),
    "3": Profile("3", "audio all", ("hello", "horn", "shouting")),
    "4": Profile("4", "cameras all", ("drowsiness", "road_sign", "lane_crossing")),
    "5": Profile("5", "crash fusion", ("crash_audio", "crash_imu")),
    "6": Profile("6", "tamper only", ("tamper",)),
    "7": Profile("7", "health/startup check only", ("health",)),
    "8": Profile("8", "harsh braking only", ("harsh",)),
    "9": Profile("9", "IMU lane changing only", ("lane",)),
    "10": Profile("10", "aggressive driving only", ("aggressive",)),
    "11": Profile("11", "hello detection only", ("hello",)),
    "12": Profile("12", "horn detection only", ("horn",)),
    "13": Profile("13", "shouting detection only", ("shouting",)),
    "14": Profile("14", "drowsiness only", ("drowsiness",)),
    "15": Profile("15", "road-sign detection only", ("road_sign",)),
    "16": Profile("16", "crash audio only", ("crash_audio",)),
    "17": Profile("17", "crash IMU only", ("crash_imu",)),
    "18": Profile("18", "GPS/LTE/backend connectivity check", ("connectivity",)),
    "19": Profile("19", "heartbeat/alive sender only", ("heartbeat",)),
    "20": Profile(
        "20",
        "audio+drowsiness+tamper+checks+alive",
        (
            "health",
            "connectivity",
            "hello",
            "horn",
            "shouting",
            "drowsiness",
            "tamper",
            "heartbeat",
            "gps_speeding",
        ),
    ),
    "21": Profile("21", "lane crossing violation only", ("lane_crossing",)),
    "22": Profile("22", "GPS-only speeding monitor", ("gps_speeding",)),
}


PROFILE_ALIASES: dict[str, str] = {
    "all": "1",
    "full": "1",
    "imu": "2",
    "imu_all": "2",
    "audio": "3",
    "audio_all": "3",
    "cameras": "4",
    "camera": "4",
    "crash": "5",
    "crash_fusion": "5",
    "tamper": "6",
    "health": "7",
    "check": "7",
    "harsh": "8",
    "lane": "9",
    "aggressive": "10",
    "hello": "11",
    "horn": "12",
    "shouting": "13",
    "shout": "13",
    "drowsiness": "14",
    "drowsy": "14",
    "road_sign": "15",
    "roadsign": "15",
    "sign": "15",
    "crash_audio": "16",
    "crash_imu": "17",
    "crash-imu": "17",
    "connectivity": "18",
    "gps_lte": "18",
    "heartbeat": "19",
    "alive": "19",
    "demo_core": "20",
    "audio_drowsy": "20",
    "audio_drowsiness": "20",
    "presentation": "20",
    "lane_crossing": "21",
    "lane-crossing": "21",
    "lane_cross": "21",
    "lane_violation": "21",
    "solid_line": "21",
    "gps_speeding": "22",
    "gps-speeding": "22",
    "gps_speed": "22",
    "overspeed": "22",
}


def normalize_token(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def resolve_models(profile: str | None, models: str | None) -> tuple[str, ...]:
    selected: list[str] = []

    if profile:
        for raw in profile.split(","):
            key = normalize_token(raw)
            key = PROFILE_ALIASES.get(key, key)
            if key in PROFILES:
                selected.extend(PROFILES[key].models)
            elif key in MODEL_ALIASES:
                selected.append(MODEL_ALIASES[key])
            else:
                raise ValueError(f"Unknown profile/model: {raw}")

    if models:
        for raw in models.split(","):
            key = normalize_token(raw)
            if key not in MODEL_ALIASES:
                raise ValueError(f"Unknown model: {raw}")
            selected.append(MODEL_ALIASES[key])

    deduped: list[str] = []
    for item in selected:
        if item not in deduped:
            deduped.append(item)
    return tuple(deduped)


def profile_menu() -> str:
    lines = ["Numeric profiles:"]
    for number in sorted(PROFILES, key=lambda value: int(value)):
        profile = PROFILES[number]
        models = ", ".join(profile.models)
        lines.append(f"  {number:>2}  {profile.label:<36} {models}")
    lines.append("")
    lines.append("Examples:")
    lines.append("  python -m demo_runtime.demo --profile 1 --api-base-url https://<tunnel>.trycloudflare.com")
    lines.append("  python -m demo_runtime.demo --profile hello")
    lines.append("  python -m demo_runtime.demo --models hello,horn,shouting")
    lines.append("  python -m demo_runtime.demo --models harsh,aggressive,tamper")
    return "\n".join(lines)

"""Step 4 - the crossing decision state machine.

Consumes the per-frame solid-line position (from line_tracker, a float in [0,1] or
None) and decides whether a real crossing happened. It is deliberately conservative
to avoid false violation events:

  * Hysteresis zones: a line counts as "left" only below HYSTERESIS_LEFT and
    "right" only above HYSTERESIS_RIGHT. The dead band in between is ignored, so
    jitter around center never flips the decision.
  * Temporal confirmation: after the line leaves its committed side, it must reach
    and HOLD the opposite zone for CONFIRM_FRAMES frames (tolerating dead-band /
    no-line frames in between) before a crossing is declared.
  * Direction: left_to_right or right_to_left.
  * Confidence: how consistently a solid line was actually visible during the move.
  * Cooldown: after firing, suppress new events for COOLDOWN_FRAMES so one physical
    crossing logs exactly once.

No video and no torch needed to test this - it is pure numbers. See main() for
scripted self-tests.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import config_crossing as cfg


@dataclass
class CrossingEvent:
    frame_index: int
    direction: str        # "left_to_right" | "right_to_left"
    confidence: float     # 0..1, detection consistency during the crossing
    position: float       # tracked position at the moment of confirmation


@dataclass
class CrossingDetector:
    left_zone: float = cfg.HYSTERESIS_LEFT
    right_zone: float = cfg.HYSTERESIS_RIGHT
    confirm_frames: int = cfg.CONFIRM_FRAMES
    cooldown_frames: int = cfg.COOLDOWN_FRAMES
    max_jump: float = cfg.MAX_POSITION_JUMP
    jump_confirm_frames: int = cfg.JUMP_CONFIRM_FRAMES
    require_center_passage: bool = cfg.REQUIRE_CENTER_PASSAGE

    # internal state
    committed_zone: str | None = None     # "left" | "right": where the line last rested
    pending_zone: str | None = None       # opposite zone being accumulated
    pending_count: int = 0
    cooldown: int = 0
    frame_index: int = -1
    prev_position: float | None = None    # last accepted position, for the jump guard
    saw_center: bool = False              # passed through the dead band since committing
    jump_candidate: float | None = None   # outlier position awaiting confirmation
    jump_candidate_count: int = 0
    _detect_window: deque = field(default_factory=lambda: deque(maxlen=8))

    def _hard_zone(self, position: float) -> str | None:
        if position < self.left_zone:
            return "left"
        if position > self.right_zone:
            return "right"
        return None  # dead band

    def _confidence(self) -> float:
        if not self._detect_window:
            return 0.0
        return round(sum(self._detect_window) / len(self._detect_window), 2)

    def update(self, position: float | None) -> CrossingEvent | None:
        """Feed one frame's tracked position. Returns a CrossingEvent or None."""
        self.frame_index += 1
        self._detect_window.append(1 if position is not None else 0)
        if self.cooldown > 0:
            self.cooldown -= 1

        if position is None:
            # no line this frame: hold state and coast (keep prev_position)
            return None

        # Jump guard with 1-frame-spike tolerance. A real line moves gradually; a big
        # jump is either a transient noise spike (ignore it) or a genuine switch to a
        # DIFFERENT line (re-baseline, never fire). We tell them apart by persistence.
        if self.prev_position is not None and abs(position - self.prev_position) > self.max_jump:
            if self.jump_candidate is not None and abs(position - self.jump_candidate) <= self.max_jump:
                self.jump_candidate_count += 1
            else:
                self.jump_candidate, self.jump_candidate_count = position, 1

            if self.jump_candidate_count >= self.jump_confirm_frames:
                # persistent -> genuine line-switch; re-baseline without firing
                self.committed_zone = self._hard_zone(position)
                self.pending_zone, self.pending_count = None, 0
                self.saw_center = False
                self.prev_position = position
                self.jump_candidate, self.jump_candidate_count = None, 0
            # transient outlier this frame: hold state, don't update prev_position
            return None

        # accepted as continuous motion
        self.jump_candidate, self.jump_candidate_count = None, 0
        self.prev_position = position

        # In the dead band (center region): the line is passing through the middle.
        if self.left_zone <= position <= self.right_zone:
            self.saw_center = True
            return None

        zone = "left" if position < self.left_zone else "right"
        event: CrossingEvent | None = None

        if self.committed_zone is None:
            self.committed_zone = zone
            self.pending_zone, self.pending_count = None, 0
            self.saw_center = False
        elif zone == self.committed_zone:
            # back to the resting side - cancel any pending crossing
            self.pending_zone, self.pending_count = None, 0
            self.saw_center = False
        else:
            if self.pending_zone == zone:
                self.pending_count += 1
            else:
                self.pending_zone, self.pending_count = zone, 1

            if self.pending_count >= self.confirm_frames:
                center_ok = self.saw_center or not self.require_center_passage
                if self.cooldown == 0 and center_ok:
                    event = CrossingEvent(
                        frame_index=self.frame_index,
                        direction=f"{self.committed_zone}_to_{zone}",
                        confidence=self._confidence(),
                        position=float(position),
                    )
                    self.cooldown = self.cooldown_frames
                # commit either way so we don't re-fire every subsequent frame
                self.committed_zone = zone
                self.pending_zone, self.pending_count = None, 0
                self.saw_center = False
        return event

    def state(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "committed_zone": self.committed_zone,
            "pending_zone": self.pending_zone,
            "pending_count": self.pending_count,
            "cooldown": self.cooldown,
        }


def process_sequence(positions, **kwargs) -> list[CrossingEvent]:
    """Run a detector over a list of positions; return all events fired."""
    detector = CrossingDetector(**kwargs)
    events = [detector.update(p) for p in positions]
    return [e for e in events if e is not None]


# --- scripted self-tests ---------------------------------------------------
def _scenarios():
    left = cfg.HYSTERESIS_LEFT
    right = cfg.HYSTERESIS_RIGHT
    middle = (left + right) / 2.0
    left_rest = max(0.02, left - 0.06)
    right_rest = min(0.98, right + 0.06)
    left_near = max(0.01, left - 0.02)
    right_near = min(0.99, right + 0.02)
    left_far = max(0.01, left - 0.12)
    right_far = min(0.99, right + max(cfg.MAX_POSITION_JUMP + 0.05, 0.20))
    confirm_right = [right_near + min(i, 3) * 0.005 for i in range(max(cfg.CONFIRM_FRAMES, 4))]
    confirm_left = [left_near - min(i, 3) * 0.005 for i in range(max(cfg.CONFIRM_FRAMES, 4))]
    return {
        "real_crossing_left_to_right": (
            [left_rest, left_rest + 0.01, left_rest, left_near]  # resting left
            + [left + 0.01, middle, right - 0.01]                 # sliding through the dead band
            + confirm_right,                                      # held in right zone -> fire once
            1,
        ),
        "teleport_flip_no_crossing": (           # the event-1 phantom: jump between lines
            [left_far, left_far, left_far]       # tracking the left edge
            + [right_far] * 5,                   # nearest-center flips to the right edge
            0,                                   # never passed center -> rejected
        ),
        "jitter_no_crossing": (
            [left_rest, left_rest + 0.01, left_rest]
            + [right_near, middle, right_near + 0.01, middle, right_near, middle],  # flapping, never held
            0,
        ),
        "deadband_wobble": (
            [middle - 0.02, middle + 0.02, middle - 0.01, middle + 0.01] * 2,  # all inside dead band
            0,
        ),
        "crossing_with_dropouts": (
            [left_rest, left_rest, left_near]
            + [None, middle, right - 0.01, None]     # passes center, briefly lost
            + [right_near, None] + confirm_right,    # still confirms (dropouts tolerated)
            1,
        ),
        "two_crossings_with_cooldown": (
            [left_rest, left_rest, left + 0.01, middle, right - 0.01]  # slide through center...
            + confirm_right                                            # fire 1: left->right
            + [right_rest] * (cfg.COOLDOWN_FRAMES + 2)                 # wait out cooldown on the right
            + [right - 0.01, middle, left + 0.01]                      # slide back through center...
            + confirm_left,                                            # fire 2: right->left
            2,
        ),
    }


def main() -> int:
    print("Crossing state-machine self-tests")
    print(f"  zones: left<{cfg.HYSTERESIS_LEFT}  right>{cfg.HYSTERESIS_RIGHT}  "
          f"confirm={cfg.CONFIRM_FRAMES} frames  cooldown={cfg.COOLDOWN_FRAMES}\n")

    all_ok = True
    for name, (positions, expected) in _scenarios().items():
        events = process_sequence(positions)
        ok = len(events) == expected
        all_ok &= ok
        flag = "OK " if ok else "FAIL"
        print(f"[{flag}] {name}: expected {expected} event(s), got {len(events)}")
        for e in events:
            print(f"        -> frame {e.frame_index}: {e.direction} (conf={e.confidence})")

    print("\nAll scenarios passed." if all_ok else "\nSome scenarios FAILED.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

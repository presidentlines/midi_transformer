"""Convert MIDI files to model tokens and back again.

The representation uses four tokens per note::

    TIME_<shift> PITCH_<pitch> DURATION_<duration> VELOCITY_<bucket>

Times and durations are quantized to quarter-second steps. This deliberately
keeps the vocabulary small and fixed, at the cost of some timing precision.
"""

from pathlib import Path
from typing import Iterable

import pretty_midi


TIME_STEPS_PER_SECOND = 4  # One time step is 0.25 seconds.
MAX_TIME_SHIFT = 64  # Maximum represented gap is 16 seconds.
MAX_DURATION = 64  # Maximum represented note duration is 16 seconds.
VELOCITY_BUCKETS = 8

SPECIAL_TOKENS = ("PAD", "BOS", "EOS")


def quantize_time(seconds: float) -> int:
    """Convert seconds to the nearest non-negative time step."""
    return max(0, round(seconds * TIME_STEPS_PER_SECOND))


def quantize_duration(seconds: float) -> int:
    """Convert a duration to a bounded number of time steps."""
    return min(MAX_DURATION, max(1, quantize_time(seconds)))


def quantize_velocity(velocity: int) -> int:
    """Convert a MIDI velocity (0--127) to a bounded bucket index."""
    bucket_width = 128 // VELOCITY_BUCKETS
    bucket = velocity // bucket_width
    return min(VELOCITY_BUCKETS - 1, max(0, bucket))


def midi_to_tokens(path: str | Path) -> list[str]:
    """Load a MIDI file and return its quantized note-event tokens."""
    midi = pretty_midi.PrettyMIDI(str(path))
    events = []

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            events.append(
                {
                    "start": note.start,
                    "pitch": note.pitch,
                    "duration": note.end - note.start,
                    "velocity": note.velocity,
                }
            )

    # The secondary pitch sort makes simultaneous notes deterministic.
    events.sort(key=lambda event: (event["start"], event["pitch"]))

    tokens = ["BOS"]
    previous_start = 0.0

    for event in events:
        time_shift = quantize_time(event["start"] - previous_start)
        time_shift = min(MAX_TIME_SHIFT, time_shift)

        tokens.extend(
            [
                f"TIME_{time_shift}",
                f"PITCH_{event['pitch']}",
                f"DURATION_{quantize_duration(event['duration'])}",
                f"VELOCITY_{quantize_velocity(event['velocity'])}",
            ]
        )
        previous_start = event["start"]

    tokens.append("EOS")
    return tokens


def build_vocabulary() -> tuple[dict[str, int], dict[int, str]]:
    """Create deterministic token-to-ID and ID-to-token mappings."""
    tokens = list(SPECIAL_TOKENS)
    tokens.extend(f"TIME_{value}" for value in range(MAX_TIME_SHIFT + 1))
    tokens.extend(f"PITCH_{pitch}" for pitch in range(128))
    tokens.extend(f"DURATION_{duration}" for duration in range(1, MAX_DURATION + 1))
    tokens.extend(f"VELOCITY_{bucket}" for bucket in range(VELOCITY_BUCKETS))

    token_to_id = {token: index for index, token in enumerate(tokens)}
    id_to_token = {index: token for token, index in token_to_id.items()}
    return token_to_id, id_to_token


def encode_tokens(tokens: Iterable[str], token_to_id: dict[str, int]) -> list[int]:
    """Convert named tokens to the integer IDs consumed by a model."""
    return [token_to_id[token] for token in tokens]


def decode_ids(token_ids: Iterable[int], id_to_token: dict[int, str]) -> list[str]:
    """Convert model token IDs back to named tokens."""
    return [id_to_token[int(token_id)] for token_id in token_ids]


def tokens_to_midi(
    tokens: Iterable[str], output_path: str | Path | None = None
) -> pretty_midi.PrettyMIDI:
    """Decode valid four-token note groups into a single-track piano MIDI.

    Invalid or out-of-order generated tokens are skipped so partially valid
    model output can still be inspected and evaluated.
    """
    tokens = list(tokens)
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name="Generated Piano")
    current_start = 0.0
    index = 0

    while index < len(tokens):
        token = tokens[index]

        if token in {"PAD", "BOS"}:
            index += 1
            continue
        if token == "EOS":
            break

        group = tokens[index : index + 4]
        if len(group) == 4 and (
            group[0].startswith("TIME_")
            and group[1].startswith("PITCH_")
            and group[2].startswith("DURATION_")
            and group[3].startswith("VELOCITY_")
        ):
            try:
                time_steps = int(group[0].split("_", maxsplit=1)[1])
                pitch = int(group[1].split("_", maxsplit=1)[1])
                duration_steps = int(group[2].split("_", maxsplit=1)[1])
                velocity_bucket = int(group[3].split("_", maxsplit=1)[1])
            except ValueError:
                index += 1
                continue

            # Validate values too, since generated strings need not come from
            # this module's vocabulary.
            if not (
                0 <= time_steps <= MAX_TIME_SHIFT
                and 0 <= pitch <= 127
                and 1 <= duration_steps <= MAX_DURATION
                and 0 <= velocity_bucket < VELOCITY_BUCKETS
            ):
                index += 1
                continue

            current_start += time_steps / TIME_STEPS_PER_SECOND
            duration = duration_steps / TIME_STEPS_PER_SECOND
            bucket_width = 128 // VELOCITY_BUCKETS
            velocity = velocity_bucket * bucket_width + bucket_width // 2
            velocity = min(127, max(1, velocity))

            instrument.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=current_start,
                    end=current_start + duration,
                )
            )
            index += 4
        else:
            index += 1

    midi.instruments.append(instrument)

    if output_path is not None:
        midi.write(str(output_path))

    return midi

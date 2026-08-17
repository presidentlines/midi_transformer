"""
Find midi files
Load midi files
Dataset related utilities
"""

from pathlib import Path
import random

import pretty_midi
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.tokenizer import build_vocabulary, encode_tokens, midi_to_tokens


def get_midi_files(data_dir="data/musicgenerationdataset", debug=True):

    data_dir = Path(data_dir)
    midi_files = list(data_dir.rglob("*.mid")) + list(data_dir.rglob("*.midi"))

    if debug:
        print(f"Found {len(midi_files)} MIDI files")

        for path in midi_files[:10]:
            print(path)

    return midi_files


def get_midi_note_df(midi_files):

    rows = []

    for path in midi_files:
        midi = load_midi(path)

        for instrument_number, instrument in enumerate(midi.instruments):
            for note in instrument.notes:
                rows.append(
                    {
                        "file": path.name,
                        "instrument": instrument_number,
                        "pitch": note.pitch,
                        "velocity": note.velocity,
                        "start": note.start,
                        "end": note.end,
                        "duration": note.end - note.start,
                    }
                )

    notes_df = pd.DataFrame(rows)

    return notes_df


def print_example_file(midi_files):

    sample_path = midi_files[0]
    midi = load_midi(sample_path)

    print("File:", sample_path.name)
    print("Duration:", midi.get_end_time(), "seconds")
    print("Estimated tempo:", midi.estimate_tempo())
    print("Number of instruments:", len(midi.instruments))

    for instrument in midi.instruments:
        print(
            instrument.name,
            "program:",
            instrument.program,
            "notes:",
            len(instrument.notes),
        )


def load_midi(path):
    return pretty_midi.PrettyMIDI(str(path))


def split_midi_files(
    midi_files,
    train_fraction=0.75,
    validation_fraction=0.125,
    seed=42,
):
    """Split whole MIDI files before tokenization to prevent data leakage."""
    midi_files = list(midi_files)

    if not midi_files:
        raise ValueError("No MIDI files were provided")
    if train_fraction <= 0 or validation_fraction < 0:
        raise ValueError("Split fractions must be non-negative and train must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError(
            "train_fraction + validation_fraction must be less than 1"
        )

    rng = random.Random(seed)
    rng.shuffle(midi_files)

    number_of_files = len(midi_files)
    train_end = round(number_of_files * train_fraction)
    validation_end = train_end + round(
        number_of_files * validation_fraction
    )

    train_files = midi_files[:train_end]
    validation_files = midi_files[train_end:validation_end]
    test_files = midi_files[validation_end:]

    if not validation_files or not test_files:
        raise ValueError("The dataset is too small for the requested split fractions")

    return train_files, validation_files, test_files


def make_token_windows(token_ids, context_length=256, stride=None):
    """Create fixed-length input/target pairs for next-token prediction."""
    if context_length <= 0:
        raise ValueError("context_length must be positive")

    if stride is None:
        stride = context_length
    if stride <= 0:
        raise ValueError("stride must be positive")

    windows = []

    # Each sequence needs one extra token so targets can be shifted by one.
    for start in range(0, len(token_ids) - context_length, stride):
        sequence = token_ids[start : start + context_length + 1]

        if len(sequence) == context_length + 1:
            windows.append((sequence[:-1], sequence[1:]))

    return windows


class MidiTokenDataset(Dataset):
    """In-memory collection of token windows created song by song."""

    def __init__(
        self,
        midi_files,
        token_to_id,
        context_length=256,
        stride=None,
    ):
        self.windows = []

        for path in midi_files:
            tokens = midi_to_tokens(path)
            token_ids = encode_tokens(tokens, token_to_id)
            self.windows.extend(
                make_token_windows(
                    token_ids,
                    context_length=context_length,
                    stride=stride,
                )
            )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        inputs, targets = self.windows[index]
        return (
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
        )


def create_datasets(
    data_dir="data/musicgenerationdataset",
    context_length=256,
    stride=None,
    seed=42,
):
    """Create reproducible train, validation, and test MIDI datasets."""
    midi_files = get_midi_files(data_dir, debug=False)
    train_files, validation_files, test_files = split_midi_files(
        midi_files,
        seed=seed,
    )
    token_to_id, id_to_token = build_vocabulary()

    datasets = {
        "train": MidiTokenDataset(
            train_files,
            token_to_id,
            context_length=context_length,
            stride=stride,
        ),
        "validation": MidiTokenDataset(
            validation_files,
            token_to_id,
            context_length=context_length,
            stride=stride,
        ),
        "test": MidiTokenDataset(
            test_files,
            token_to_id,
            context_length=context_length,
            stride=stride,
        ),
        "train_files": train_files,
        "validation_files": validation_files,
        "test_files": test_files,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
    }
    return datasets


def create_dataloaders(datasets, batch_size=16):
    """Wrap each dataset in a DataLoader suitable for training/evaluation."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=batch_size,
            shuffle=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
        ),
    }

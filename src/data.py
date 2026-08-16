"""
Find midi files
Load midi files
Dataset related utilities
"""

from pathlib import Path
import pretty_midi
import pandas as pd


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

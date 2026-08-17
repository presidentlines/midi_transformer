import pretty_midi
import data as midi_data
import matplotlib.pyplot as plt


def visualize_piano_roll(sample_path):

    midi = pretty_midi.PrettyMIDI(str(sample_path))
    piano_roll = midi.get_piano_roll(fs=10)

    plt.figure(figsize=(12, 5))
    plt.imshow(piano_roll[:, :300], origin="lower", aspect="auto")

    plt.title(f"First 30 Seconds of {sample_path.name}")
    plt.xlabel("Time")
    plt.ylabel("MIDI Pitch")
    plt.show()

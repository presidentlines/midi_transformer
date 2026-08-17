"""Evaluate the trained transformer and compare real and generated MIDI."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data import MidiTokenDataset
from src.model import MidiTransformer
from src.tokenizer import encode_tokens, midi_to_tokens


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = MidiTransformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return checkpoint, model


def split_paths(checkpoint, split, data_dir):
    paths = [data_dir / name for name in checkpoint["file_splits"][split]]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing MIDI file: {missing[0]}")
    return paths


@torch.no_grad()
def test_loss(model, loader, device):
    total_loss = 0.0
    total_tokens = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        total_loss += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="sum",
        ).item()
        total_tokens += targets.numel()
    return total_loss / total_tokens


def bigram_loss(train_paths, test_loader, token_to_id):
    """Calculate test loss for an add-one-smoothed bigram model."""
    vocabulary_size = len(token_to_id)
    counts = torch.ones((vocabulary_size, vocabulary_size), dtype=torch.float64)

    for path in train_paths:
        token_ids = encode_tokens(midi_to_tokens(path), token_to_id)
        previous = torch.tensor(token_ids[:-1])
        following = torch.tensor(token_ids[1:])
        counts.index_put_(
            (previous, following),
            torch.ones(len(previous), dtype=torch.float64),
            accumulate=True,
        )

    log_probabilities = torch.log(counts / counts.sum(dim=1, keepdim=True))
    total_loss = 0.0
    total_tokens = 0
    for inputs, targets in test_loader:
        total_loss -= log_probabilities[
            inputs.reshape(-1), targets.reshape(-1)
        ].sum().item()
        total_tokens += targets.numel()
    return total_loss / total_tokens


def load_notes(paths):
    notes = []
    for path in paths:
        midi = pretty_midi.PrettyMIDI(str(path))
        for instrument in midi.instruments:
            if not instrument.is_drum:
                notes.extend(instrument.notes)
    return notes


def note_statistics(notes):
    pitches = np.array([note.pitch for note in notes])
    durations = np.array([note.end - note.start for note in notes])
    return {
        "number_of_notes": len(notes),
        "mean_pitch": float(pitches.mean()),
        "pitch_standard_deviation": float(pitches.std()),
        "mean_duration": float(durations.mean()),
        "unique_pitches": int(np.unique(pitches).size),
        "pitch_range": [int(pitches.min()), int(pitches.max())],
    }


def plot_training(history_path, output_dir):
    history = json.loads(history_path.read_text())
    epochs = [row["epoch"] for row in history]
    training = [row["training_loss"] for row in history]
    validation = [row["validation_loss"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, training, label="Training loss")
    plt.plot(epochs, validation, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training and Validation Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=200)
    plt.close()


def plot_perplexity(transformer_perplexity, bigram_perplexity, output_dir):
    labels = ["Transformer", "Bigram baseline"]
    values = [transformer_perplexity, bigram_perplexity]
    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, values, color=["#2878B5", "#C82423"])
    plt.bar_label(bars, fmt="%.2f")
    plt.ylabel("Held-out test perplexity")
    plt.title("Model Comparison (Lower Is Better)")
    plt.tight_layout()
    plt.savefig(output_dir / "perplexity_comparison.png", dpi=200)
    plt.close()


def plot_distributions(real_notes, generated_notes, output_dir):
    real_pitches = [note.pitch for note in real_notes]
    generated_pitches = [note.pitch for note in generated_notes]
    plt.figure(figsize=(9, 4))
    plt.hist(real_pitches, bins=range(128), density=True, alpha=0.6, label="Test MIDI")
    plt.hist(
        generated_pitches, bins=range(128), density=True, alpha=0.6,
        label="Generated MIDI",
    )
    plt.xlabel("MIDI pitch")
    plt.ylabel("Density")
    plt.title("Pitch Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pitch_distribution.png", dpi=200)
    plt.close()

    real_durations = [note.end - note.start for note in real_notes]
    generated_durations = [note.end - note.start for note in generated_notes]
    maximum = float(np.percentile(real_durations, 99))
    bins = np.linspace(0, maximum, 41)
    plt.figure(figsize=(9, 4))
    plt.hist(real_durations, bins=bins, density=True, alpha=0.6, label="Test MIDI")
    plt.hist(
        generated_durations, bins=bins, density=True, alpha=0.6,
        label="Generated MIDI",
    )
    plt.xlabel("Individual note duration (seconds)")
    plt.ylabel("Density")
    plt.title("Note-Duration Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "duration_distribution.png", dpi=200)
    plt.close()


def plot_piano_roll(real_path, generated_path, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, sharey=True)
    for axis, path, title in zip(
        axes,
        [real_path, generated_path],
        ["Held-out Test Example", "Generated Example"],
    ):
        midi = pretty_midi.PrettyMIDI(str(path))
        roll = midi.get_piano_roll(fs=10)[:, :300]
        axis.imshow(
            roll,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            extent=[0, roll.shape[1] / 10, 0, 128],
        )
        axis.set_ylabel("MIDI pitch")
        axis.set_title(f"{title}: {path.name}")
    axes[-1].set_xlabel("Time (seconds)")
    fig.tight_layout()
    fig.savefig(output_dir / "piano_roll_examples.png", dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_model.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/musicgenerationdataset"))
    parser.add_argument("--generated-dir", type=Path, default=Path("artifacts/generated/many"))
    parser.add_argument("--history", type=Path, default=Path("artifacts/training_history.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    generated_paths = sorted(args.generated_dir.glob("*.mid"))
    if not generated_paths:
        raise FileNotFoundError(f"No MIDI files found in {args.generated_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint, model = load_model(args.checkpoint, device)
    train_paths = split_paths(checkpoint, "train", args.data_dir)
    test_paths = split_paths(checkpoint, "test", args.data_dir)
    test_dataset = MidiTokenDataset(
        test_paths,
        checkpoint["token_to_id"],
        context_length=checkpoint["model_config"]["context_length"],
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Evaluating on {device}...")
    transformer_loss = test_loss(model, test_loader, device)
    baseline_loss = bigram_loss(train_paths, test_loader, checkpoint["token_to_id"])
    transformer_perplexity = math.exp(transformer_loss)
    baseline_perplexity = math.exp(baseline_loss)

    real_notes = load_notes(test_paths)
    generated_notes = load_notes(generated_paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_training(args.history, args.output_dir)
    plot_perplexity(transformer_perplexity, baseline_perplexity, args.output_dir)
    plot_distributions(real_notes, generated_notes, args.output_dir)
    representative = max(
        generated_paths,
        key=lambda path: pretty_midi.PrettyMIDI(str(path)).get_end_time(),
    )
    plot_piano_roll(test_paths[0], representative, args.output_dir)

    results = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "transformer_test_loss": transformer_loss,
        "transformer_test_perplexity": transformer_perplexity,
        "bigram_test_loss": baseline_loss,
        "bigram_test_perplexity": baseline_perplexity,
        "real_test_notes": note_statistics(real_notes),
        "generated_notes": note_statistics(generated_notes),
        "generated_files": [path.name for path in generated_paths],
    }
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    print(f"Transformer test perplexity: {transformer_perplexity:.3f}")
    print(f"Bigram test perplexity: {baseline_perplexity:.3f}")
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()

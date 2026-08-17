"""Evaluate a trained MIDI transformer and save metrics and plots."""

import argparse
import csv
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
from src.generate import choose_device, generate_token_ids, set_seed
from src.model import MidiTransformer
from src.tokenizer import decode_ids, encode_tokens, midi_to_tokens, tokens_to_midi


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = MidiTransformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return checkpoint, model


def paths_from_checkpoint(checkpoint, split, data_dir):
    paths = [data_dir / name for name in checkpoint["file_splits"][split]]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing MIDI file: {missing[0]}")
    return paths


@torch.no_grad()
def transformer_test_loss(model, loader, device):
    total_loss = 0.0
    total_tokens = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += targets.numel()

    if total_tokens == 0:
        raise ValueError("The test split did not produce any token windows")
    return total_loss / total_tokens


def bigram_test_loss(train_files, test_loader, token_to_id):
    """Evaluate an add-one-smoothed next-token bigram baseline."""
    vocabulary_size = len(token_to_id)
    counts = torch.ones(
        (vocabulary_size, vocabulary_size),
        dtype=torch.float64,
    )

    for path in train_files:
        ids = encode_tokens(midi_to_tokens(path), token_to_id)
        previous = torch.tensor(ids[:-1], dtype=torch.long)
        following = torch.tensor(ids[1:], dtype=torch.long)
        counts.index_put_((previous, following), torch.ones_like(previous, dtype=torch.float64), accumulate=True)

    log_probabilities = torch.log(counts / counts.sum(dim=1, keepdim=True))
    total_negative_log_likelihood = 0.0
    total_tokens = 0

    for inputs, targets in test_loader:
        total_negative_log_likelihood -= log_probabilities[
            inputs.reshape(-1), targets.reshape(-1)
        ].sum().item()
        total_tokens += targets.numel()

    return total_negative_log_likelihood / total_tokens


def grammar_score(tokens):
    """Return the fraction of generated transitions following token grammar."""
    tokens = [token for token in tokens if token != "PAD"]
    if len(tokens) < 2:
        return 0.0

    def token_type(token):
        if token in {"BOS", "EOS"}:
            return token
        return token.split("_", maxsplit=1)[0]

    allowed = {
        "BOS": {"TIME", "EOS"},
        "TIME": {"PITCH"},
        "PITCH": {"DURATION"},
        "DURATION": {"VELOCITY"},
        "VELOCITY": {"TIME", "EOS"},
    }
    valid = 0
    evaluated = 0

    for current, following in zip(tokens, tokens[1:]):
        current_type = token_type(current)
        following_type = token_type(following)
        evaluated += 1
        if following_type in allowed.get(current_type, set()):
            valid += 1

    return valid / evaluated


def notes_from_midi_files(paths):
    notes = []
    for path in paths:
        midi = pretty_midi.PrettyMIDI(str(path))
        for instrument in midi.instruments:
            if not instrument.is_drum:
                notes.extend(instrument.notes)
    return notes


def note_statistics(notes):
    if not notes:
        return {
            "number_of_notes": 0,
            "mean_pitch": None,
            "pitch_standard_deviation": None,
            "mean_duration": None,
            "mean_velocity": None,
            "unique_pitches": 0,
            "pitch_range": None,
        }

    pitches = np.array([note.pitch for note in notes])
    durations = np.array([note.end - note.start for note in notes])
    velocities = np.array([note.velocity for note in notes])
    return {
        "number_of_notes": len(notes),
        "mean_pitch": float(pitches.mean()),
        "pitch_standard_deviation": float(pitches.std()),
        "mean_duration": float(durations.mean()),
        "mean_velocity": float(velocities.mean()),
        "unique_pitches": int(np.unique(pitches).size),
        "pitch_range": [int(pitches.min()), int(pitches.max())],
    }


def notes_from_midi(path):
    """Load all non-drum notes from one MIDI file."""
    midi = pretty_midi.PrettyMIDI(str(path))
    return [
        note
        for instrument in midi.instruments
        if not instrument.is_drum
        for note in instrument.notes
    ]


def piece_statistics(path):
    """Return interpretable, song-level musical features for one MIDI file."""
    notes = notes_from_midi(path)
    if not notes:
        return {
            "file": path.name,
            "number_of_notes": 0,
            "duration_seconds": 0.0,
            "notes_per_second": 0.0,
            "unique_pitches": 0,
            "pitch_range": 0,
            "mean_pitch": None,
            "pitch_standard_deviation": None,
            "mean_note_duration": None,
            "pitch_class_entropy": None,
            "mean_onset_polyphony": None,
        }

    pitches = np.array([note.pitch for note in notes])
    durations = np.array([note.end - note.start for note in notes])
    piece_duration = max(note.end for note in notes)
    pitch_class_counts = np.bincount(pitches % 12, minlength=12)
    pitch_class_probabilities = pitch_class_counts / pitch_class_counts.sum()
    nonzero_probabilities = pitch_class_probabilities[pitch_class_probabilities > 0]
    normalized_entropy = -np.sum(
        nonzero_probabilities * np.log2(nonzero_probabilities)
    ) / np.log2(12)
    onset_counts = {}
    for note in notes:
        quantized_onset = round(note.start, 6)
        onset_counts[quantized_onset] = onset_counts.get(quantized_onset, 0) + 1

    return {
        "file": path.name,
        "number_of_notes": len(notes),
        "duration_seconds": float(piece_duration),
        "notes_per_second": float(len(notes) / piece_duration) if piece_duration else 0.0,
        "unique_pitches": int(np.unique(pitches).size),
        "pitch_range": int(pitches.max() - pitches.min()),
        "mean_pitch": float(pitches.mean()),
        "pitch_standard_deviation": float(pitches.std()),
        "mean_note_duration": float(durations.mean()),
        "pitch_class_entropy": float(normalized_entropy),
        "mean_onset_polyphony": float(np.mean(list(onset_counts.values()))),
    }


def save_training_plot(history_path, output_dir):
    if not history_path.is_file():
        return None
    history = json.loads(history_path.read_text())
    if not history:
        return None

    epochs = [row["epoch"] for row in history]
    training = [row["training_loss"] for row in history]
    validation = [row["validation_loss"] for row in history]
    best_index = int(np.argmin(validation))

    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(epochs, training, linewidth=2, label="Training")
    axis.plot(epochs, validation, linewidth=2, label="Validation")
    axis.scatter(
        epochs[best_index], validation[best_index], color="black", zorder=3,
        label=f"Best validation (epoch {epochs[best_index]})",
    )
    axis.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training Convergence")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    output_path = output_dir / "training_curves.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_perplexity_plot(model_perplexity, baseline_perplexity, output_dir):
    fig, axis = plt.subplots(figsize=(6.5, 4.8))
    labels = ["Transformer", "Bigram baseline"]
    values = [model_perplexity, baseline_perplexity]
    bars = axis.bar(labels, values, color=["#2878B5", "#C82423"], width=0.62)
    axis.bar_label(bars, fmt="%.2f", padding=4)
    axis.set(ylabel="Held-out test perplexity", title="Next-Token Prediction Performance")
    axis.set_ylim(0, max(values) * 1.18)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "perplexity_comparison.png", dpi=200)
    plt.close(fig)


def save_pitch_class_plot(real_notes, generated_notes, output_dir):
    labels = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]

    def proportions(notes):
        counts = np.bincount([note.pitch % 12 for note in notes], minlength=12)
        return counts / counts.sum() if counts.sum() else counts.astype(float)

    real = proportions(real_notes)
    generated = proportions(generated_notes)
    positions = np.arange(12)
    width = 0.42
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(positions - width / 2, real, width, label="Held-out test", color="#2878B5")
    axis.bar(positions + width / 2, generated, width, label="Generated", color="#F8AC3D")
    axis.set(
        xticks=positions,
        xticklabels=labels,
        xlabel="Pitch class",
        ylabel="Proportion of notes",
        title="Pitch-Class Distribution",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "pitch_class_distribution.png", dpi=200)
    plt.close(fig)


def save_piece_feature_plot(real_rows, generated_rows, output_dir):
    features = [
        ("notes_per_second", "Note density\n(notes/s)"),
        ("unique_pitches", "Unique\npitches"),
        ("mean_note_duration", "Mean duration\n(seconds)"),
        ("pitch_class_entropy", "Pitch-class\nentropy"),
    ]
    fig, axes = plt.subplots(1, len(features), figsize=(13, 4.8))
    for axis, (key, label) in zip(axes, features):
        values = [
            [row[key] for row in real_rows if row[key] is not None],
            [row[key] for row in generated_rows if row[key] is not None],
        ]
        boxes = axis.boxplot(values, tick_labels=["Test", "Generated"], patch_artist=True)
        for patch, color in zip(boxes["boxes"], ["#2878B5", "#F8AC3D"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for position, group in enumerate(values, start=1):
            jitter = np.linspace(-0.07, 0.07, len(group)) if group else []
            axis.scatter(np.full(len(group), position) + jitter, group, s=24, color="black", alpha=0.65)
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Song-Level Musical Feature Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "song_feature_comparison.png", dpi=200)
    plt.close(fig)


def save_piano_roll_plot(real_path, generated_path, output_dir, seconds=30):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)
    for axis, path, label in zip(
        axes,
        [real_path, generated_path],
        ["Held-out test example", "Generated example"],
    ):
        midi = pretty_midi.PrettyMIDI(str(path))
        piano_roll = midi.get_piano_roll(fs=10)
        axis.imshow(
            piano_roll[:, : seconds * 10], origin="lower", aspect="auto",
            interpolation="nearest", cmap="magma",
            extent=[0, min(seconds, piano_roll.shape[1] / 10), 0, 128],
        )
        axis.set(ylabel="MIDI pitch", title=f"{label}: {path.name}")
    axes[-1].set_xlabel("Time (seconds)")
    fig.tight_layout()
    fig.savefig(output_dir / "piano_roll_examples.png", dpi=200)
    plt.close(fig)


def save_piece_statistics(real_paths, generated_paths, output_dir):
    real_rows = [piece_statistics(path) | {"source": "held_out_test"} for path in real_paths]
    generated_rows = [piece_statistics(path) | {"source": "generated"} for path in generated_paths]
    rows = real_rows + generated_rows
    if rows:
        with (output_dir / "piece_statistics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    save_piece_feature_plot(real_rows, generated_rows, output_dir)
    return real_rows, generated_rows


def save_distribution_plots(real_notes, generated_notes, output_dir):
    real_pitches = [note.pitch for note in real_notes]
    generated_pitches = [note.pitch for note in generated_notes]
    real_durations = [note.end - note.start for note in real_notes]
    generated_durations = [note.end - note.start for note in generated_notes]

    plt.figure(figsize=(10, 4))
    plt.hist(real_pitches, bins=range(128), density=True, alpha=0.6, label="Test MIDI")
    if generated_pitches:
        plt.hist(generated_pitches, bins=range(128), density=True, alpha=0.6, label="Generated")
    plt.xlabel("MIDI pitch")
    plt.ylabel("Density")
    plt.title("Real and Generated Pitch Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pitch_distribution.png", dpi=200)
    plt.close()

    upper_duration = max(0.25, float(np.percentile(real_durations, 99)))
    duration_bins = np.linspace(0, upper_duration, 41)
    plt.figure(figsize=(10, 4))
    plt.hist(real_durations, bins=duration_bins, density=True, alpha=0.6, label="Test MIDI")
    if generated_durations:
        plt.hist(generated_durations, bins=duration_bins, density=True, alpha=0.6, label="Generated")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Density")
    plt.title("Real and Generated Duration Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "duration_distribution.png", dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_model.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/musicgenerationdataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=None,
        help="Analyze existing .mid files instead of generating new evaluation samples",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("artifacts/training_history.json"),
        help="Training history JSON used for the convergence figure",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.samples <= 0:
        raise ValueError("samples must be positive")

    device = choose_device()
    checkpoint, model = load_checkpoint(args.checkpoint, device)
    token_to_id = checkpoint["token_to_id"]
    id_to_token = checkpoint["id_to_token"]
    train_files = paths_from_checkpoint(checkpoint, "train", args.data_dir)
    test_files = paths_from_checkpoint(checkpoint, "test", args.data_dir)

    test_dataset = MidiTokenDataset(
        test_files,
        token_to_id,
        context_length=checkpoint["model_config"]["context_length"],
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Using device: {device}")
    print("Evaluating held-out test windows...")
    model_loss = transformer_test_loss(model, test_loader, device)
    baseline_loss = bigram_test_loss(train_files, test_loader, token_to_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_notes = []
    generated_paths = []
    sample_results = []

    if args.generated_dir is not None:
        generated_paths = sorted(args.generated_dir.glob("*.mid"))
        if not generated_paths:
            raise FileNotFoundError(f"No MIDI files found in {args.generated_dir}")
        print(f"Analyzing {len(generated_paths)} existing generated samples...")
        for output_path in generated_paths:
            notes = notes_from_midi(output_path)
            generated_notes.extend(notes)
            sample_results.append(
                {
                    "number_of_valid_notes": len(notes),
                    "midi_file": str(output_path),
                }
            )
    else:
        print(f"Generating {args.samples} evaluation samples...")
        for sample_number in range(args.samples):
            seed = args.seed + sample_number
            set_seed(seed)
            token_ids = generate_token_ids(
                model,
                token_to_id,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device,
            )
            tokens = decode_ids(token_ids, id_to_token)
            output_path = args.output_dir / f"generated_seed_{seed}.mid"
            midi = tokens_to_midi(tokens, output_path)
            notes = [note for instrument in midi.instruments for note in instrument.notes]
            generated_notes.extend(notes)
            generated_paths.append(output_path)
            sample_results.append(
                {
                    "seed": seed,
                    "number_of_tokens": len(tokens),
                    "number_of_valid_notes": len(notes),
                    "grammar_rate": grammar_score(tokens),
                    "midi_file": output_path.name,
                }
            )

    real_notes = notes_from_midi_files(test_files)
    save_distribution_plots(real_notes, generated_notes, args.output_dir)
    save_pitch_class_plot(real_notes, generated_notes, args.output_dir)
    real_piece_rows, generated_piece_rows = save_piece_statistics(
        test_files, generated_paths, args.output_dir
    )
    representative_generated = max(
        generated_paths,
        key=lambda path: piece_statistics(path)["duration_seconds"],
    )
    save_piano_roll_plot(test_files[0], representative_generated, args.output_dir)
    save_training_plot(args.history, args.output_dir)
    save_perplexity_plot(math.exp(model_loss), math.exp(baseline_loss), args.output_dir)

    if args.generated_dir is not None:
        sampling = {
            "source": "existing_files",
            "directory": str(args.generated_dir),
            "note": "Generation settings cannot be inferred from MIDI files.",
        }
    else:
        sampling = {
            "source": "generated_by_evaluator",
            "temperature": args.temperature,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
        }

    results = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_loss": checkpoint.get("validation_loss"),
        "transformer_test_loss": model_loss,
        "transformer_test_perplexity": math.exp(model_loss),
        "bigram_test_loss": baseline_loss,
        "bigram_test_perplexity": math.exp(baseline_loss),
        "sampling": sampling,
        "real_test_notes": note_statistics(real_notes),
        "generated_notes": note_statistics(generated_notes),
        "real_test_piece_statistics": real_piece_rows,
        "generated_piece_statistics": generated_piece_rows,
        "temporally_collapsed_samples": [
            row["file"]
            for row in generated_piece_rows
            if row["duration_seconds"] <= 1.0 or row["mean_onset_polyphony"] > 4
        ],
        "generated_samples": sample_results,
    }
    results_path = args.output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")

    print(f"Transformer test perplexity: {results['transformer_test_perplexity']:.3f}")
    print(f"Bigram test perplexity: {results['bigram_test_perplexity']:.3f}")
    print(f"Saved evaluation results to {args.output_dir}")


if __name__ == "__main__":
    main()

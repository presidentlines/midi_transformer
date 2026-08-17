"""Evaluate a trained MIDI transformer and save metrics and plots."""

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
    plt.savefig(output_dir / "pitch_distribution.png", dpi=150)
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
    plt.savefig(output_dir / "duration_distribution.png", dpi=150)
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
    sample_results = []

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

    results = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_loss": checkpoint.get("validation_loss"),
        "transformer_test_loss": model_loss,
        "transformer_test_perplexity": math.exp(model_loss),
        "bigram_test_loss": baseline_loss,
        "bigram_test_perplexity": math.exp(baseline_loss),
        "sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
        },
        "real_test_notes": note_statistics(real_notes),
        "generated_notes": note_statistics(generated_notes),
        "generated_samples": sample_results,
    }
    results_path = args.output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")

    print(f"Transformer test perplexity: {results['transformer_test_perplexity']:.3f}")
    print(f"Bigram test perplexity: {results['bigram_test_perplexity']:.3f}")
    print(f"Saved evaluation results to {args.output_dir}")


if __name__ == "__main__":
    main()

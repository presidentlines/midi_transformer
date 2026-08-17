"""Generate a MIDI file from a trained transformer checkpoint."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from src.model import MidiTransformer
from src.tokenizer import decode_ids, tokens_to_midi


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path, device):
    """Reconstruct a model and its vocabulary from a training checkpoint."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model = MidiTransformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return (
        model,
        checkpoint["token_to_id"],
        checkpoint["id_to_token"],
    )


def midi_to_wav(midi, output_path, sample_rate=44_100):
    """Render a simple synthesized WAV file for convenient playback."""
    has_notes = any(instrument.notes for instrument in midi.instruments)
    if has_notes:
        audio = midi.synthesize(fs=sample_rate)
    else:
        audio = np.zeros(sample_rate, dtype=np.float64)

    audio = np.nan_to_num(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak

    wavfile.write(
        output_path,
        sample_rate,
        (audio * np.iinfo(np.int16).max).astype(np.int16),
    )


@torch.no_grad()
def generate_token_ids(
    model,
    token_to_id,
    max_tokens=512,
    min_tokens=0,
    temperature=0.9,
    top_k=20,
    device=torch.device("cpu"),
):
    """Autoregressively sample token IDs beginning with BOS."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if min_tokens < 0 or min_tokens > max_tokens:
        raise ValueError("min_tokens must be between 0 and max_tokens")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    generated = torch.tensor(
        [[token_to_id["BOS"]]],
        dtype=torch.long,
        device=device,
    )
    eos_id = token_to_id["EOS"]

    for token_number in range(max_tokens):
        # The model can only attend to its configured context length.
        context = generated[:, -model.context_length :]
        next_token_logits = model(context)[:, -1, :] / temperature
        if token_number < min_tokens:
            next_token_logits[:, eos_id] = -torch.inf

        number_of_candidates = min(top_k, next_token_logits.size(-1))
        top_values, top_indices = torch.topk(
            next_token_logits,
            k=number_of_candidates,
            dim=-1,
        )
        probabilities = torch.softmax(top_values, dim=-1)
        sampled_position = torch.multinomial(probabilities, num_samples=1)
        next_token = top_indices.gather(-1, sampled_position)

        generated = torch.cat((generated, next_token), dim=1)
        if next_token.item() == eos_id:
            break

    return generated.squeeze(0).tolist()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/best_model.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/generated/sample.mid"),
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=128,
        help="Do not allow EOS before this many tokens are generated",
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of samples to generate using consecutive seeds",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Train the model first with "
            "`uv run python -m src.train --epochs 10`."
        )
    if args.samples <= 0:
        raise ValueError("samples must be positive")

    device = choose_device()
    print(f"Using device: {device}")

    model, token_to_id, id_to_token = load_model(args.checkpoint, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for sample_number in range(args.samples):
        seed = args.seed + sample_number
        set_seed(seed)

        if args.samples == 1:
            midi_output = args.output
        else:
            midi_output = args.output.with_name(
                f"{args.output.stem}_seed_{seed}{args.output.suffix}"
            )

        token_ids = generate_token_ids(
            model,
            token_to_id,
            max_tokens=args.max_tokens,
            min_tokens=args.min_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        tokens = decode_ids(token_ids, id_to_token)
        midi = tokens_to_midi(tokens, midi_output)
        wav_output = midi_output.with_suffix(".wav")
        midi_to_wav(midi, wav_output)
        number_of_notes = sum(
            len(instrument.notes) for instrument in midi.instruments
        )

        print(
            f"Seed {seed}: generated {len(tokens)} tokens and "
            f"{number_of_notes} valid notes"
        )
        print(f"  MIDI: {midi_output}")
        print(f"  WAV:  {wav_output}")


if __name__ == "__main__":
    main()

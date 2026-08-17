"""Generate a MIDI file from a trained transformer checkpoint."""

import argparse
import random
from pathlib import Path

import torch

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


@torch.no_grad()
def generate_token_ids(
    model,
    token_to_id,
    max_tokens=512,
    temperature=0.9,
    top_k=20,
    device=torch.device("cpu"),
):
    """Autoregressively sample token IDs beginning with BOS."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
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

    for _ in range(max_tokens):
        # The model can only attend to its configured context length.
        context = generated[:, -model.context_length :]
        next_token_logits = model(context)[:, -1, :] / temperature

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
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Train the model first with "
            "`uv run python -m src.train --epochs 10`."
        )

    set_seed(args.seed)
    device = choose_device()
    print(f"Using device: {device}")

    model, token_to_id, id_to_token = load_model(args.checkpoint, device)
    token_ids = generate_token_ids(
        model,
        token_to_id,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    tokens = decode_ids(token_ids, id_to_token)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    midi = tokens_to_midi(tokens, args.output)
    number_of_notes = sum(
        len(instrument.notes) for instrument in midi.instruments
    )

    print(f"Generated {len(tokens)} tokens and {number_of_notes} valid notes")
    print(f"Saved MIDI to {args.output}")


if __name__ == "__main__":
    main()

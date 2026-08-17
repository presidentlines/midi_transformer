"""Train the MIDI transformer from the command line."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from src.data import create_dataloaders, create_datasets
from src.model import MidiTransformer


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


def next_token_loss(logits, targets):
    """Calculate cross-entropy across every token position."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


def train_one_epoch(model, loader, optimizer, device, log_every=10):
    if log_every < 0:
        raise ValueError("log_every must be non-negative")

    model.train()
    total_loss = 0.0
    started_at = time.monotonic()

    for batch_number, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = next_token_loss(model(inputs), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if log_every > 0 and (
            batch_number % log_every == 0 or batch_number == len(loader)
        ):
            elapsed = time.monotonic() - started_at
            seconds_per_batch = elapsed / batch_number
            remaining = seconds_per_batch * (len(loader) - batch_number)
            print(
                f"  batch {batch_number:>3}/{len(loader)}: "
                f"running_loss={total_loss / batch_number:.4f}, "
                f"elapsed={elapsed:.0f}s, eta={remaining:.0f}s",
                flush=True,
            )

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        total_loss += next_token_loss(model(inputs), targets).item()

    return total_loss / len(loader)


def overfit_one_batch(model, loader, device, steps=100, learning_rate=3e-4):
    """Sanity-check the pipeline by repeatedly learning one fixed batch."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    inputs, targets = next(iter(loader))
    inputs = inputs.to(device)
    targets = targets.to(device)
    losses = []

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = next_token_loss(model(inputs), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

        if step == 1 or step % 10 == 0 or step == steps:
            print(f"step {step:>4}/{steps}: loss={loss.item():.4f}")

    return losses


def save_checkpoint(path, model, model_config, datasets, epoch, validation_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "token_to_id": datasets["token_to_id"],
            "id_to_token": datasets["id_to_token"],
            "epoch": epoch,
            "validation_loss": validation_loss,
            "file_splits": {
                "train": [path.name for path in datasets["train_files"]],
                "validation": [
                    path.name for path in datasets["validation_files"]
                ],
                "test": [path.name for path in datasets["test_files"]],
            },
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data/musicgenerationdataset",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print batch progress every N batches; use 0 to disable",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/best_model.pt"),
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("artifacts/training_history.json"),
    )
    parser.add_argument("--overfit-one-batch", action="store_true")
    parser.add_argument("--overfit-steps", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device()
    print(f"Using device: {device}", flush=True)

    print("Loading and tokenizing MIDI files...", flush=True)
    datasets = create_datasets(
        data_dir=args.data_dir,
        context_length=args.context_length,
        seed=args.seed,
    )
    loaders = create_dataloaders(datasets, batch_size=args.batch_size)
    print(
        "Files: "
        f"{len(datasets['train_files'])} train, "
        f"{len(datasets['validation_files'])} validation, "
        f"{len(datasets['test_files'])} test"
    )
    print(
        "Windows: "
        f"{len(datasets['train'])} train, "
        f"{len(datasets['validation'])} validation, "
        f"{len(datasets['test'])} test"
    )

    model_config = {
        "vocabulary_size": len(datasets["token_to_id"]),
        "context_length": args.context_length,
        "embedding_dimension": 128,
        "number_of_heads": 4,
        "number_of_layers": 3,
        "feedforward_dimension": 512,
        "dropout": 0.1,
    }
    model = MidiTransformer(**model_config).to(device)

    if args.overfit_one_batch:
        losses = overfit_one_batch(
            model,
            loaders["train"],
            device,
            steps=args.overfit_steps,
            learning_rate=args.learning_rate,
        )
        print(f"Loss changed from {losses[0]:.4f} to {losses[-1]:.4f}")
        return

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    history = []
    best_validation_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        epoch_started_at = time.monotonic()
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        training_loss = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            log_every=args.log_every,
        )
        print("  validating...", flush=True)
        validation_loss = evaluate(model, loaders["validation"], device)
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"epoch {epoch:>3}/{args.epochs}: "
            f"train_loss={training_loss:.4f}, "
            f"validation_loss={validation_loss:.4f}, "
            f"time={time.monotonic() - epoch_started_at:.0f}s",
            flush=True,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(
                args.checkpoint,
                model,
                model_config,
                datasets,
                epoch,
                validation_loss,
            )
            print(f"Saved best checkpoint to {args.checkpoint}")

        # Persist progress after every epoch so an interrupted run still has
        # usable loss history.
        args.history.parent.mkdir(parents=True, exist_ok=True)
        args.history.write_text(json.dumps(history, indent=2) + "\n")

    print(f"Saved training history to {args.history}")


if __name__ == "__main__":
    main()

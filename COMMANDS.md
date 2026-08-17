# MIDI Transformer Commands

Run these commands from the repository root.

## Pull changes and install dependencies

```bash
git pull --ff-only
uv sync
```

## Check the available PyTorch device

```bash
uv run python -c "import torch; print('MPS built:', torch.backends.mps.is_built()); print('MPS available:', torch.backends.mps.is_available()); print('CUDA available:', torch.cuda.is_available())"
```

## Run the one-batch sanity check

```bash
uv run python -m src.train --overfit-one-batch
```

## Train for 10 epochs without batch messages

```bash
uv run python -m src.train --epochs 10 --log-every 0
```

## Train for 10 epochs with progress every 5 batches

```bash
uv run python -m src.train --epochs 10 --log-every 5
```

## Confirm that training artifacts exist

```bash
ls -lh artifacts/best_model.pt artifacts/training_history.json
```

## Generate one MIDI sample

```bash
uv run python -m src.generate \
  --checkpoint artifacts/best_model.pt \
  --output artifacts/generated/sample_42.mid \
  --temperature 0.9 \
  --top-k 20 \
  --max-tokens 512 \
  --seed 42
```

## Generate three reproducible MIDI samples

```bash
uv run python -m src.generate --seed 42 --output artifacts/generated/sample_42.mid
uv run python -m src.generate --seed 43 --output artifacts/generated/sample_43.mid
uv run python -m src.generate --seed 44 --output artifacts/generated/sample_44.mid
```

## Run the full evaluation

```bash
uv run python -m src.evaluate \
  --checkpoint artifacts/best_model.pt \
  --output-dir artifacts/evaluation
```

## Inspect evaluation results

```bash
uv run python -m json.tool artifacts/evaluation/results.json
ls -lh artifacts/evaluation
```

## Open generated files on macOS

```bash
open artifacts/generated/sample_42.mid
open artifacts/evaluation/pitch_distribution.png
open artifacts/evaluation/duration_distribution.png
```

## Open generated files on Linux

```bash
xdg-open artifacts/generated/sample_42.mid
xdg-open artifacts/evaluation/pitch_distribution.png
xdg-open artifacts/evaluation/duration_distribution.png
```

## Check repository changes

```bash
git status --short
git diff --check
```

## Commit source and result files

```bash
git add README.md COMMANDS.md src pyproject.toml uv.lock
git add artifacts/training_history.json artifacts/evaluation artifacts/generated
git status --short
git commit -m "Complete MIDI transformer training and evaluation"
git push
```

## Commit the final checkpoint if it is intentionally tracked

```bash
git add -f artifacts/best_model.pt
git commit -m "Add trained MIDI transformer checkpoint"
git push
```

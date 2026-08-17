# MIDI Transformer Commands

## Train a model

```bash
uv run python -m src.train --epochs 10 --log-every 0
```

## Generate music

```bash
uv run python -m src.generate --checkpoint artifacts/best_model.pt --output artifacts/generated/sample.mid --seed 42
```

## Evaluate the model

```bash
uv run python -m src.evaluate --checkpoint artifacts/best_model.pt --output-dir artifacts/evaluation
```

## View the evaluation results

```bash
uv run python -m json.tool artifacts/evaluation/results.json
```

# MIDI Transformer Commands

## Train a model

```bash
uv run python -m src.train --epochs 10 --log-every 0
```

## Generate music

```bash
uv run python -m src.generate --checkpoint artifacts/best_model.pt --output artifacts/generated/sample.mid --samples 5 --max-tokens 256 --temperature 0.7 --top-k 10 --seed 42
```

## Evaluate the model

```bash
uv run python -m src.evaluate --checkpoint artifacts/best_model.pt --output-dir artifacts/evaluation
```

## View the evaluation results

```bash
uv run python -m json.tool artifacts/evaluation/results.json
```

## Generate 10 different songs

```bash
uv run python -m src.generate --checkpoint artifacts/best_model.pt --output artifacts/generated/many/candidates.mid --seed 42 --samples 10 --max-tokens 256 --temperature .7 --top-k 10
```

# play all the songs in a folder
```bash
for file in artifacts/generated/many/*.wav; do
    echo "$file"
    afplay "$file"
done
```
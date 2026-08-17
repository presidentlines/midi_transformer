# MIDI Transformer

This project trains a small decoder-only transformer to generate symbolic piano
music from MIDI files. The model learns next-token prediction over note events
that describe timing, pitch, duration, and velocity.

The project uses a collection of 32 classical MIDI files containing music by
Bach, Beethoven, and other composers. MIDI is used instead of audio because it
provides structured note and timing information directly. Music files come from this kaggle dataset: https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi


## Setup

The project requires Python 3.13 or newer and uses
[uv](https://docs.astral.sh/uv/) for dependency management.

From the project directory, install the locked dependencies:

```bash
uv sync
```

The MIDI dataset is already located in:

```text
data/musicgenerationdataset/
```

## Sanity-check training

Before a full experiment, verify that the model can overfit one fixed batch:

```bash
uv run python -m src.train --overfit-one-batch
```

The loss should decrease substantially. 
## Train the model

Run the full training experiment with:

```bash
uv run python -m src.train --epochs 10 --log-every 5
```

The 32 songs are split by file into 24 training, 4 validation, and 4 test
pieces. Splitting by song prevents windows from the same composition from
appearing in both training and evaluation data.

Training saves out two things:

```text
artifacts/best_model.pt
artifacts/training_history.json
```

`best_model.pt` contains the weights from the epoch with the lowest validation
loss. Progress is logged during each epoch with running loss and estimated
remaining time.

## Generate MIDI

After training has produced `artifacts/best_model.pt`, generate a sample with:

```bash
uv run python -m src.generate \
  --output artifacts/generated/sample_01.mid \
  --temperature 0.9 \
  --top-k 20 \
  --seed 42
```

Generation begins with `BOS` and samples one token at a time until the model
produces `EOS` or reaches the maximum token count. Temperature controls
randomness, while top-k sampling limits choices to likely tokens.

Generate reproducible variations by changing the seed:

```bash
uv run python -m src.generate --seed 43 \
  --output artifacts/generated/sample_43.mid
uv run python -m src.generate --seed 44 \
  --output artifacts/generated/sample_44.mid
```

## Evaluation

After training, run the held-out evaluation with:

```bash
uv run python -m src.evaluate
```

This saves test loss and perplexity, a bigram baseline, generated-token grammar
rates, statistics, generated MIDI examples, and plots
under `artifacts/evaluation/`.

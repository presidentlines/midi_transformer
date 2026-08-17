# MIDI Transformer

This project trains a small decoder-only transformer to generate symbolic piano
music from MIDI files. The model learns next-token prediction over note events
that describe timing, pitch, duration, and velocity.

The project uses a collection of 32 classical MIDI files containing music by
Bach, Beethoven, and other composers. MIDI is used instead of audio because it
provides structured note and timing information directly.

## Project pipeline

```text
MIDI files
    -> quantized event tokens
    -> song-level train/validation/test split
    -> causal transformer
    -> generated event tokens
    -> generated MIDI file
```

Each note is represented by four tokens:

```text
TIME_<shift> PITCH_<pitch> DURATION_<duration> VELOCITY_<bucket>
```

Timing and duration are quantized into 0.25-second steps. The fixed vocabulary
contains 268 tokens, including `BOS`, `EOS`, and `PAD`.

## Model

The default model is a small causal transformer designed for the limited
dataset size:

- 256-token context length
- 128-dimensional embeddings
- 3 transformer layers
- 4 attention heads
- 512-dimensional feed-forward layers
- 0.1 dropout
- approximately 697,000 parameters

The causal attention mask prevents the model from examining future tokens while
predicting the next token.

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

The loss should decrease substantially. This checks the tokenizer, dataset,
causal mask, loss calculation, backpropagation, and optimizer. It does not
produce a useful final checkpoint.

## Train the model

Run the full training experiment with:

```bash
uv run python -m src.train --epochs 10 --log-every 5
```

The 32 songs are split by file into 24 training, 4 validation, and 4 test
pieces. Splitting by song prevents windows from the same composition from
appearing in both training and evaluation data.

Training saves:

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

## Repository structure

```text
data/                 MIDI training data
docs/                 project rubric and paper planning documents
notebooks/            exploratory data analysis
src/data.py            MIDI loading and PyTorch datasets
src/tokenizer.py       MIDI tokenization and reconstruction
src/model.py           causal transformer model
src/train.py           sanity check and full training loop
src/generate.py        autoregressive MIDI generation
src/evaluate.py        evaluation work in progress
src/viz.py             exploratory visualization helpers
```

## Current limitations

- The dataset contains only 32 compositions.
- Timing is measured in seconds rather than musical beats, making the
  representation sensitive to tempo.
- Timing, duration, and velocity are quantized and therefore lose precision.
- Generated token order is not constrained; malformed note groups are skipped
  when converting output to MIDI.
- Quantitative evaluation and baseline comparisons are still in progress.

## Planned evaluation

The final evaluation will include held-out test loss and perplexity, valid token
grammar rate, comparisons of real and generated pitch and duration
distributions, generated MIDI examples, and a simple statistical baseline.

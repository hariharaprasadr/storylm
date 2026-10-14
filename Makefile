.PHONY: help story test web info train train-modal audit corpus corpus-join vocab matrix-toy narration clean

# Use the project venv when it exists; "python" is not on PATH everywhere.
PY ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || (command -v python3 || echo python))

help:
	@echo "StoryLM - Tiny Language Model"
	@echo ""
	@echo "Targets:"
	@echo "  make story   - Generate a bedtime story from the newest checkpoint"
	@echo "  make corpus  - Build the current-vocabulary training corpus and pack it"
	@echo "  make corpus-join - Rebuild train.jsonl from the committed shards"
	@echo "  make vocab   - Rebuild data/vocab.json (the current table)"
	@echo "  make test    - Run KV-cache parity and performance test"
	@echo "  make web     - Launch the interactive architecture & walkthrough visualizer"
	@echo "  make info    - Print the token tables, checkpoints and datasets"
	@echo "  make train   - Smoke training run, local, plumbing only (300 steps)"
	@echo "  make train-modal - Real training on a Modal T4 (volumes, run, download)"
	@echo "  make audit   - Overfitting / memorisation audit of a checkpoint"
	@echo "  make matrix-toy - Regenerate the web matrix walkthrough's 4-dim toy block"
	@echo "  make narration - Render every narration tier (words, hot lines, composed units)"
	@echo "  make clean   - Clean cache files"

# The current generation: data/vocab.json (3,509 ids) + data/processed/train_v4.npz.
corpus:
	$(PY) storylm/corpus_builder.py --variants 23 --target-vocab 3450 \
		--out-corpus data/processed/train.jsonl --out-vocab data/vocab.json --out-series data/series.json
	$(PY) training/prepare_data.py --corpus data/processed/train.jsonl \
		--vocab data/vocab.json --out data/processed/train_v4.npz

corpus-join:
	gunzip -c data/processed/corpus/train_part*.jsonl.gz > data/processed/train.jsonl
	@echo "joined $$(wc -l < data/processed/train.jsonl) stories into data/processed/train.jsonl"

vocab:
	$(PY) storylm/corpus_builder.py --variants 1 --target-vocab 3450

story:
	$(PY) cli.py story --world bluey --mode bedtime

test:
	$(PY) cli.py test

web:
	$(PY) cli.py web

info:
	$(PY) cli.py info

train:
	$(PY) training/train_model.py --data data/processed/train_v4.npz --steps 300

# Real training runs on a Modal T4 - never locally (see training/README.md).
# Override the dataset with DATA=train_v3.npz / VOCAB=vocab_v3.json, the run name with NAME=...
# The checkpoint records the table it was trained on, so sampling, export and
# evaluation pair it back automatically.
train-modal:
	set -a; . ./.env.local; set +a; \
	$(PY) -m modal volume create tinystory-data || true; \
	$(PY) -m modal volume create tinystory-ckpts || true; \
	$(PY) -m modal volume put --force tinystory-data data/processed/$(or $(DATA),train_v4.npz) /$(or $(DATA),train_v4.npz); \
	$(PY) -m modal volume put --force tinystory-data data/$(or $(VOCAB),vocab_v4.json) /$(or $(VOCAB),vocab_v4.json); \
	$(PY) -m modal run training/modal_train.py --steps $(or $(STEPS),5400) --batch $(or $(BATCH),64) \
		--name $(or $(NAME),embedded_v4) --data-file $(or $(DATA),train_v4.npz) --vocab-file $(or $(VOCAB),vocab_v4.json); \
	$(PY) -m modal volume get tinystory-ckpts $(or $(NAME),embedded_v4).pt checkpoints/$(or $(NAME),embedded_v4).pt

audit:
	$(PY) training/memorisation.py

# The matrix walkthrough page runs a 4-dimension slice of the shipped checkpoint.
# Regenerate it (and its reference numbers) after training a new model.
matrix-toy:
	$(PY) training/export_matrix_toy.py --search
	$(PY) training/export_matrix_toy.py

# Narration clips: rendered offline with Kokoro-82M (needs .venv-tts, python 3.13 —
# see docs/narration.md), then packed into the two blobs the website streams.
PY_TTS ?= .venv-tts/bin/python
narration:
	$(PY_TTS) training/render_narration.py --what words --out /tmp/narration
	$(PY_TTS) training/render_narration.py --what lines --out /tmp/narration \
		--lines data/narration_lines.txt
	$(PY_TTS) training/render_compose.py --stage plan   --out /tmp/narration_compose
	$(PY_TTS) training/render_compose.py --stage render --out /tmp/narration_compose
	$(PY_TTS) training/render_compose.py --stage cut    --out /tmp/narration_compose
	$(PY_TTS) training/pack_narration.py --words /tmp/narration/words \
		--lines /tmp/narration/lines --line-keys data/narration_lines.txt \
		--compose /tmp/narration_compose/units --out web/audio

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +

.PHONY: help story test web info train train-modal audit corpus corpus-join vocab clean

# Use the project venv when it exists; "python" is not on PATH everywhere.
PY ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || (command -v python3 || echo python))

help:
	@echo "StoryLM 1.67M Tiny Language Model"
	@echo ""
	@echo "Targets:"
	@echo "  make story   - Generate a bedtime story from checkpoints/embedded_big.pt"
	@echo "  make corpus  - Build 3.5K vocab training corpus and pack to .npz"
	@echo "  make corpus-join - Rebuild train.jsonl from the committed shards"
	@echo "  make vocab   - Rebuild data/vocab.json"
	@echo "  make test    - Run KV-cache parity and performance test"
	@echo "  make web     - Launch the interactive architecture & walkthrough visualizer"
	@echo "  make info    - Print model checkpoint and dataset overview"
	@echo "  make train   - Smoke training run, local, plumbing only (300 steps)"
	@echo "  make train-modal - Real training on a Modal T4 (volumes, run, download)"
	@echo "  make audit   - Overfitting / memorisation audit of the shipped checkpoint"
	@echo "  make clean   - Clean cache files"

corpus:
	$(PY) storylm/corpus_builder.py --variants 23 --target-vocab 3530
	$(PY) training/prepare_data.py --corpus data/processed/train.jsonl --vocab data/vocab.json --out data/processed/train.npz
	$(PY) training/prepare_data.py --corpus data/processed/train_big.jsonl --vocab data/vocab.json --out data/processed/train_big.npz

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
	$(PY) training/train_model.py --data data/processed/train.npz --steps 300

# Real training runs on a Modal T4 - never locally (see training/README.md).
# Override the dataset with DATA=train.npz / VOCAB=vocab.json, the run name with NAME=...
train-modal:
	set -a; . ./.env.local; set +a; \
	$(PY) -m modal volume create tinystory-data || true; \
	$(PY) -m modal volume create tinystory-ckpts || true; \
	$(PY) -m modal volume put tinystory-data data/processed/$(or $(DATA),train_v2.npz) /$(or $(DATA),train_v2.npz); \
	$(PY) -m modal volume put tinystory-data data/$(or $(VOCAB),vocab_v2.json) /$(or $(VOCAB),vocab_v2.json); \
	$(PY) -m modal run training/modal_train.py --steps $(or $(STEPS),5400) --batch $(or $(BATCH),64) \
		--name $(or $(NAME),embedded_v2) --data-file $(or $(DATA),train_v2.npz) --vocab-file $(or $(VOCAB),vocab_v2.json); \
	$(PY) -m modal volume get tinystory-ckpts $(or $(NAME),embedded_v2).pt checkpoints/$(or $(NAME),embedded_v2).pt

audit:
	$(PY) training/memorisation.py --checkpoint checkpoints/embedded.pt \
		--corpus data/processed/train.jsonl --packed data/processed/train.npz

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +

.PHONY: help story test web trace info train clean

help:
	@echo "TinyLM 1.67M Nano Language Model"
	@echo ""
	@echo "Targets:"
	@echo "  make story   - Generate a bedtime story from checkpoints/embedded_big.pt"
	@echo "  make test    - Run KV-cache parity and performance test"
	@echo "  make web     - Launch the interactive architecture & walkthrough visualizer"
	@echo "  make trace   - Regenerate web/trace_data.js forward pass trace"
	@echo "  make info    - Print model checkpoint and dataset overview"
	@echo "  make train   - Run a 300-step smoke training run"
	@echo "  make clean   - Clean cache files"

story:
	python cli.py story --world bluey --mode bedtime

test:
	python cli.py test

web:
	python cli.py web

trace:
	python cli.py trace

info:
	python cli.py info

train:
	python training/train_model.py --data data/processed/train.npz --steps 300

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +

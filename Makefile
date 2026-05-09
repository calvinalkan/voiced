MAKEFLAGS += -j$(shell nproc) --output-sync=target

.PHONY: check lint format fix typecheck typecheck-ty typecheck-pyright test test-integration test-transcribers test-realtime

check: lint typecheck test

lint:
	uv run ruff check .

format:
	uv run ruff format .

fix:
	uv run ruff check . --fix

typecheck: typecheck-ty typecheck-pyright

typecheck-ty:
	uv run ty check .

typecheck-pyright:
	uv run basedpyright .

test: test-transcribers test-integration

test-transcribers:
	uv run python test_transcriber.py

test-integration:
	./test.sh

# Slow real-time-paced tests; not in `make check` (see test_transcriber.py
# docstring for what these measure vs. the default batched tests).
test-realtime:
	uv run python test_transcriber.py --realtime

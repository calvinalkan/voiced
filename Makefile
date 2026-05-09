MAKEFLAGS += -j$(shell nproc) --output-sync=target

.PHONY: check lint format fix typecheck typecheck-ty typecheck-pyright test test-integration test-transcribers

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

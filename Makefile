MAKEFLAGS += -j$(shell nproc) --output-sync=target

.PHONY: check lint format fix typecheck typecheck-ty typecheck-pyright

check: lint typecheck

lint:
	-uv run ruff check .

format:
	uv run ruff format .

fix:
	uv run ruff check . --fix

typecheck: typecheck-ty typecheck-pyright

typecheck-ty:
	-uv run ty check .

typecheck-pyright:
	-uv run basedpyright .

uv lock --check
uv sync --locked --group dev

uv run ruff check . --fix
uv run mypy --no-incremental

uv run pytest --collect-only -q
uv run pytest -q
.PHONY: run

run:
	uv run uvicorn src.main:main_app --host 0.0.0.0 --port 9999

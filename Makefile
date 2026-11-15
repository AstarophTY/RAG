TEMP_FILE := *.egg-info build dist .pytest_cache .mypy_cache
MYPY_FLAG := --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
SRC_DIR := src
TEST_DIR := tests
UV := uv
MAKEFLAGS += -s

all: run

run: install
	$(UV) run python -m $(SRC_DIR)

install:
	$(UV) sync
	ollama pull qwen3:0.6b

.PHONY: data index test

data:
	test -d data/raw/vllm-0.10.1 || (cd data/raw && unzip -q vllm-0.10.1.zip)

index: data
	$(UV) run python -m $(SRC_DIR) index

test:
	$(UV) run pytest $(TEST_DIR)

clean:
	$(RM) -rf $(TEMP_FILE)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete

debug:
	$(UV) run python -m pdb -m src

lint:
	$(UV) run flake8 $(SRC_DIR)/**/*.py
	$(UV) run mypy $(SRC_DIR)/**/*.py $(MYPY_FLAG)

lint-strict:
	$(UV) run flake8 $(SRC_DIR)/**/*.py
	$(UV) run mypy $(SRC_DIR)/**/*.py --strict
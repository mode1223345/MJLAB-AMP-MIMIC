.PHONY: sync
sync:
	uv sync --all-packages --extra cu128 --group dev

.PHONY: sync-cpu
sync-cpu:
	uv sync --all-packages --extra cpu --group dev

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: type
type:
	uv run ty check
	uv run pyright

.PHONY: stubs
stubs:
	bash typings/generate_mujoco_stubs.sh

.PHONY: check
check: format type

.PHONY: test
test:
	uv run pytest -m "not golden"

.PHONY: test-fast
test-fast:
	uv run pytest -m "not slow and not golden"

.PHONY: test-golden
test-golden:
	uv run pytest -m golden

.PHONY: test-cpu
test-cpu:
	FORCE_CPU=1 uv run pytest -m "not golden"

.PHONY: test-cpu-fast
test-cpu-fast:
	FORCE_CPU=1 uv run pytest -m "not slow and not golden"

.PHONY: test-all
test-all: check test

.PHONY: build
build:
	uv build
	uv run --isolated --no-project --with dist/*.whl tests/smoke_test.py
	uv run --isolated --no-project --with dist/*.tar.gz tests/smoke_test.py
	@echo "Build and import test successful"

.PHONY: publish-test
publish-test: build
	uv publish --publish-url https://test.pypi.org/legacy/

.PHONY: publish
publish: build
	uv publish

.PHONY: docker-build
docker-build:
	docker build -t mjlab:latest .

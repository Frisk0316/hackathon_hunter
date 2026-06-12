.PHONY: setup dev test-unit test-integration test-frontend test-all lint frontend-check docs-check docs-impact smoke verify verify-full

setup:
	@echo "TODO: add setup command for this repo"

dev:
	@echo "TODO: add dev command for this repo"

test-unit:
	@echo "TODO: add unit test command for this repo"

test-integration:
	@echo "TODO: add integration test command for this repo"

test-frontend:
	@echo "TODO: add frontend test command for this repo"

test-all: test-unit test-integration test-frontend

lint:
	@echo "TODO: add lint command for this repo"

frontend-check:
	@echo "TODO: add frontend static check command for this repo"

docs-check:
	@echo "TODO: add docs-check command for this repo"

docs-impact:
	@echo "TODO: add docs-impact command for this repo"

smoke:
	@echo "TODO: add smoke checks for this repo"

verify: docs-check docs-impact lint test-unit
	@echo "Verification complete."

verify-full: verify test-integration test-frontend smoke
	@echo "Full verification complete."

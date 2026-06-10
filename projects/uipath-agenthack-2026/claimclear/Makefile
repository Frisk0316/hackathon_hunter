PYTHON ?= python3
PYTHONPATH := src

.PHONY: demo run-clean run-ambiguous test lint smoke qa

demo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m claimclear.cli demo

run-clean:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m claimclear.cli run --claim-id CLM-1001

run-ambiguous:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m claimclear.cli run --claim-id CLM-1002 --resolve-human approve

test:
	PYTHONPATH=$(PYTHONPATH) pytest -q

lint:
	PYTHONPATH=$(PYTHONPATH) ruff check src tests

smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m claimclear.cli demo --audit-path artifacts/smoke_audit.jsonl

qa: test lint smoke


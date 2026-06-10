.PHONY: test lint collect-mock rank-mock rules-mock

test:
	python3 -m pytest

lint:
	python3 -m ruff check .

collect-mock:
	python3 -m hackathon_hunter collect --mock

rank-mock:
	python3 -m hackathon_hunter rank --input data/processed/mock_hackathons.json

rules-mock:
	python3 -m hackathon_hunter check-rules --input data/processed/mock_hackathons.json

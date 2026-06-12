.PHONY: test lint docs-check docs-impact verify collect-mock rank-mock rules-mock status-mock watch-mock

test:
	python3 -m pytest

lint:
	python3 -m ruff check .

docs-check:
	python3 scripts/docs/check_doc_metadata.py --paths docs tasks
	python3 scripts/docs/check_feature_map_links.py

docs-impact:
	python3 scripts/docs/check_doc_impact.py

verify: lint test docs-check docs-impact

collect-mock:
	python3 -m hackathon_hunter collect --mock

rank-mock:
	python3 -m hackathon_hunter rank --input data/processed/mock_hackathons.json

rules-mock:
	python3 -m hackathon_hunter check-rules --input data/processed/mock_hackathons.json

status-mock:
	python3 -m hackathon_hunter status --input data/processed/mock_hackathons.json

watch-mock:
	python3 -m hackathon_hunter watch --input data/processed/mock_hackathons.json

.PHONY: ingest extract_json_sample extract_json_full

ingest:
	python run_integrity_check.py

extract_json_sample:
	python json_recon.py

extract_json_full:
	@echo "Full JSON extraction not yet implemented."
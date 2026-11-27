.PHONY: reproduce test

PYTHON ?= python

reproduce:
	PYTHON=$(PYTHON) bash run_all.sh

test:
	$(PYTHON) -m pytest tests/

# Flawline: dataset, evaluation and dashboard. Run from the repository root.
#   make setup      install pinned dependencies and the spaCy model
#   make baselines  download VCTK p318/p227/p376/p283/p248/p326/p345/p314 (CC BY 4.0) and the JFK excerpts (public domain), align them
#   make dataset    generate every flawed / conditioned clip, then the manifest
#   make eval       harness self-test, acceptance tests, leakage audit, headline metrics on train and dev (never the test split)
#   make app        open the dashboard on http://localhost:8501
PY ?= python
GEN = flawline-dataset/generator

setup:
	$(PY) -m pip install -r requirements.lock
	$(PY) -m spacy download en_core_web_sm

baselines:
	cd $(GEN) && $(PY) ingest_corpus.py
	bash scripts/fetch_jfk.sh

dataset:
	cd $(GEN) && $(PY) make_dataset.py --pilot B01-CHAMP --l3 --invariance --flawed-cond 40 \
	  && $(PY) make_dataset.py --grid B09-CHAMP B10-CHAMP && $(PY) make_dataset.py --pure B01-CHAMP \
	  && $(PY) make_dataset.py --invariance && $(PY) make_dataset.py --manifest && $(PY) qa.py

eval:
	$(PY) eval/test_harness.py
	$(PY) eval/acceptance.py --mode same
	$(PY) eval/leakage_audit.py
	$(PY) eval/headline.py --split train --mode same
	$(PY) eval/headline.py --split dev --mode same

app:
	streamlit run dashboard/app.py

.PHONY: setup baselines dataset eval app

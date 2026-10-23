.PHONY: install data demo run app test lint all

install:
	pip install -r requirements.txt

data:
	python -m src.data.download

demo:
	python -m src.pipeline --demo

run:
	python -m src.pipeline

app:
	streamlit run app/streamlit_app.py

test:
	python -m pytest -q

lint:
	ruff check src app tests

all: data run test

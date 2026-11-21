.PHONY: install data data-github demo run tune train figures run-full app test lint all

install:
	pip install -r requirements.txt

data:
	python -m src.data.download

data-github:            # UCI erişilemezse yedek kaynak (pip install pyreadr)
	python -m src.data.download --source online_retail_ii_github

demo:
	python -m src.pipeline --demo

run:
	python -m src.pipeline

tune:                   # doğrulama katlarında hiperparametre ayarı (~10 dk)
	python -m src.models.tune

train:                  # final modeli tüm veriyle eğit -> models/
	python -m src.models.train

figures:
	python reports/make_figures.py

run-full: run train figures

app:
	streamlit run app/streamlit_app.py

test:
	python -m pytest -q

lint:
	ruff check src app tests

all: data run-full test

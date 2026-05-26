.PHONY: install install-dev train test lint app predict

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

train:
	python src/train_model.py --real data/True.csv --fake data/Fake.csv --outdir outputs

test:
	pytest -q

lint:
	ruff check src tests

app:
	streamlit run src/streamlit_app.py

predict:
	python src/detect_fake_news.py --text "Reuters reported that lawmakers passed a new budget bill."

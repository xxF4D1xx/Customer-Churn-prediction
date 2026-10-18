.PHONY: install train explain all docker-build docker-run clean

install:
	pip install -r requirements.txt

train:
	python -m src.train

explain:
	python -m src.explain

all: train explain

docker-build:
	docker build -t churn:latest .

docker-run:
	docker run --rm \
		-v "$$(pwd)/figures:/app/figures" \
		-v "$$(pwd)/results:/app/results" \
		churn:latest

clean:
	rm -f figures/*.png results/*.json
.PHONY: help build run stop clean test migrate

help:
	@echo "Available commands:"
	@echo "  make build    - Build Docker image"
	@echo "  make run      - Run with docker-compose"
	@echo "  make stop     - Stop containers"
	@echo "  make clean    - Clean up containers and volumes"
	@echo "  make test     - Run tests"
	@echo "  make migrate  - Run database migrations"
	@echo "  make admin    - Create admin API key"

build:
	docker-compose build

run:
	docker-compose up -d
	@echo "BitTorch is running at http://localhost:8000"
	@echo "API docs at http://localhost:8000/docs"

stop:
	docker-compose down

clean:
	docker-compose down -v
	rm -rf data/*.db
	rm -rf data/models/*.pth

test:
	docker-compose run --rm bittorch pytest tests/

migrate:
	docker-compose run --rm bittorch alembic upgrade head

admin:
	docker-compose run --rm bittorch python -c "from app.auth import auth_service; from app.database import Base, engine; Base.metadata.create_all(bind=engine); key = auth_service.create_api_key('Admin', 'admin', 1000); print(f'Admin API Key: {key}')"

logs:
	docker-compose logs -f bittorch

shell:
	docker-compose exec bittorch /bin/bash
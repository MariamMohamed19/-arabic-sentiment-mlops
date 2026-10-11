# Setup

python -m venv .venv
.venv\Scripts\activate   # Linux/macOS: source .venv/bin/activate
pip install -e .
pytest
uvicorn src.api.main:app --reload
docker compose up --build

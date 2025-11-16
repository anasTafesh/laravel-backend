#  Laravel API

## Auth Endpoints

### 1. Register

- **URL:** `POST http://127.0.0.1:8000/api/register_user`
- **Body (x-www-form-urlencoded):**
  - `name`
  - `email`
  - `password`
  - `password_confirmation`

### 2. Login

- **URL:** `POST http://127.0.0.1:8000/api/login_user`
- **Body (x-www-form-urlencoded):**
  - `email`
  - `password`

### 3. Logout

- **URL:** `POST http://127.0.0.1:8000/api/logout`
- **Headers:**
  - `Authorization: Bearer <token>`
  - `Accept: application/json`
# Flask ML API (`flask-api/`)

Endpoints:
- `POST /predict` – classify emotions
- `POST /generate` – generate implicit meanings
- `POST /analyze_reviews` – full pipeline: generate → classify → summarize
- `POST /summarize_topics` – summarize a set of texts
- `GET /health` – health check

## Run locally

```bash
cd flask-api
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # fill Reddit credentials if needed
python app.py         # starts on port 5000

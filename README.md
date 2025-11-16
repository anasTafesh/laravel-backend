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

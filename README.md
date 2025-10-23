# Ultra Transcriber – Web (Ready to Deploy)

## Quick Start (Docker)
```bash
# 1) unzip the project folder
# 2) (optional) copy .env.example to .env and edit values
docker compose up -d --build
# open http://localhost:7860
# login: admin@example.com / admin123
```

## Local (without Docker)
```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn UltraTranscriberWeb:app --app-dir app --host 0.0.0.0 --port 7860 --reload
```

## Environment Variables
See `.env.example`. Use:
- BILLING_MODE=ads (default) or subscription
- For Stripe subscription mode, set STRIPE_SECRET_KEY and STRIPE_PRICE_ID
- For AdSense ads, set ADSENSE_CLIENT
- APP_SECRET, BASE_URL

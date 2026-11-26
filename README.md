# Metal Prediction

Multi-horizon quantitative commodity intelligence terminal covering **Gold (Au)**, **Silver (Ag)**, **Platinum (Pt)**, **Copper (Cu)**, and **Aluminium (Al)** across MCX (INR) and global benchmarks (USD), powered by a locally trained **MetalNLP-v1** assistant (`0` external AI API dependencies).

## Features
- **5-Metal Interactive Dashboard (`/`)**: 3D rotary metal carousel, multi-horizon (`1D`–`5Y`) forecast curves, COMEX/LME-to-MCX landed parity & duty/GST stack, arbitrage spread, Tax & Shock Simulator, FX Matrix, FinBERT sentiment feed, and Burger Dashboard drawer.
- **Custom Trained `MetalNLP-v1` Chatbot (`POST /api/chat`)**: Sublinear TF-IDF (word 1–2g + char 3–4g) softmax logistic regression classifier trained locally (`train_chatbot.js`, weights stored in `model/metal_nlp_weights.json`) with numerical parity & horizon slot extraction.
- **Compliance & Trust Architecture**: Session-gated 4-second startup skeleton (`?replaySkeleton=true`), `localStorage` cookie consent preferences, non-SEBI prediction disclaimers, support FAQ & contact form (`POST /api/contact`), offline API banner (`GET /api/health`), `/methodology`, `/disclaimer`, `/privacy`, `/terms`, `/cookies`, and custom `404.html`.
- **Zero-Cost Market Cache (`GET /api/prices`)**: 15-minute server-side TTL snapshot cache.

## Quick Start (Local)

```bash
npm install
npm start
```

Server starts at `http://localhost:3000` (and mirrors locally at `http://localhost:8765`).

To retrain `MetalNLP-v1` weights from the supervised corpus:

```bash
npm run train
```

## API Endpoints
- `GET /api/health` — Server status & loaded `MetalNLP-v1` model metrics
- `GET /api/status` — Project metadata and supported metals
- `GET /api/prices` — TTL-cached 5-metal spot & landed parity snapshot
- `POST /api/chat` — Local `MetalNLP-v1` quantitative Q&A (`{ message, sessionId, context }`)
- `POST /api/contact` — Support & feedback submission handler

## Deployment
- **Vercel (Zero-Config Express)**: `server.js` exports the Express `app` instance and skips `app.listen()` when `process.env.VERCEL` is set; `public/` and `model/metal_nlp_weights.json` are bundled automatically.
- **Node Hosts (Render / Railway / Fly)**: Run `npm start` (`node server.js`), which binds `process.env.PORT`.

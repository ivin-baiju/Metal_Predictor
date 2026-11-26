import express from "express";
import rateLimit from "express-rate-limit";
import "dotenv/config";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { getMetalChatbot } from "./chatbot_engine.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load dynamically trained real-market commodity ML models if available
let trainedCommodityModels = null;
try {
  const modelFile = path.join(__dirname, "model", "trained_commodity_models.json");
  if (fs.existsSync(modelFile)) {
    trainedCommodityModels = JSON.parse(fs.readFileSync(modelFile, "utf8"));
  }
} catch (_e) {}

const app = express();
app.set("trust proxy", 1);
const metalChatbot = getMetalChatbot();

app.use(express.json({ limit: "10kb" }));

// Serve both ./public and root workspace directory so index.html and logo.svg resolve anywhere
app.use(express.static(path.join(__dirname, "public")));
app.use(express.static(__dirname));

// Basic protection against excessive requests (15 RPM per IP)
app.use(
  "/api/chat",
  rateLimit({
    windowMs: 60 * 1000,
    limit: 15,
    standardHeaders: true,
    legacyHeaders: false,
    message: { error: "Rate limit reached (15 requests/min). Please wait a moment before asking again." }
  })
);

// In-memory conversation storage & bounded feedback queue
const sessions = new Map();
const feedbackSubmissions = [];

app.use(
  "/api/contact",
  rateLimit({
    windowMs: 15 * 60 * 1000,
    limit: 10,
    standardHeaders: true,
    legacyHeaders: false,
    message: { error: "Feedback rate limit reached (10 submissions / 15 min). Please try again shortly." }
  })
);

app.use(
  "/api/prices",
  rateLimit({
    windowMs: 60 * 1000,
    limit: 60,
    standardHeaders: true,
    legacyHeaders: false,
    message: { error: "Price cache rate limit reached (60 requests/min)." }
  })
);

// ============================================================================
// METAL PREDICT RAG KNOWLEDGE BASE (STRICTLY 5 METALS: GOLD, SILVER, PLATINUM, COPPER, ALUMINIUM)
// ============================================================================
const METAL_PREDICT_KNOWLEDGE = {
  asOfDate: "November 25, 2026",
  usdInrSpot: 92.19,
  troyOzGrams: 31.1035,
  architecture: "Hybrid Ensemble combining Temporal Fusion Transformer (TFT), N-BEATS Neural Basis, XGBoost, LightGBM Quantile Regression, and Vector Error Correction Model (VECM).",
  metals: {
    gold: {
      name: "Gold",
      symbol: "Au",
      atomicNumber: 79,
      mcxUnit: "10g",
      globalUnit: "troy oz",
      mcxSpotInr: 121086,
      globalSpotUsd: 3551.00,
      baseLandedInr: 105251,
      taxPaidParityInr: 114913,
      paritySpreadInr: 6173,
      paritySpreadPct: 5.37,
      dayChangePct: -2.85,
      dutyPct: 6.0,
      gstPct: 3.0,
      accuracy: "78.4%",
      r2: "0.912",
      maeInr: 840,
      rmseInr: 1120,
      maeUsd: 24.50,
      rmseUsd: 32.80,
      sentimentScore: "+0.42 (Bullish)",
      topDriver: "US 10Y TIPS Real Yield (31%) & Central Bank Net Purchases (20%)",
      forecasts: { "1D": "+0.62% (₹1,21,837)", "1W": "+1.84% (₹1,23,314)", "1M": "+4.43% (₹1,26,450)", "3M": "+7.85% (₹1,30,591)", "6M": "+11.40% (₹1,34,890)", "1Y": "+16.90% (₹1,41,550)", "5Y": "+62.50% (₹1,96,765)" }
    },
    silver: {
      name: "Silver",
      symbol: "Ag",
      atomicNumber: 47,
      mcxUnit: "1kg",
      globalUnit: "troy oz",
      mcxSpotInr: 143748,
      globalSpotUsd: 40.89,
      baseLandedInr: 121196,
      taxPaidParityInr: 132322,
      paritySpreadInr: 11426,
      paritySpreadPct: 8.63,
      dayChangePct: -5.74,
      dutyPct: 6.0,
      gstPct: 3.0,
      accuracy: "74.9%",
      r2: "0.884",
      maeInr: 1620,
      rmseInr: 2190,
      maeUsd: 0.46,
      rmseUsd: 0.62,
      sentimentScore: "+0.56 (Strong Bullish)",
      topDriver: "Solar PV TOPCon Paste Demand (34%) & Gold/Silver Ratio Cointegration (24%)",
      forecasts: { "1D": "+1.15% (₹1,45,401)", "1W": "+2.90% (₹1,47,917)", "1M": "+6.80% (₹1,53,523)", "3M": "+12.10% (₹1,61,141)", "6M": "+17.50% (₹1,68,904)", "1Y": "+24.40% (₹1,78,823)", "5Y": "+88.20% (₹2,70,534)" }
    },
    platinum: {
      name: "Platinum",
      symbol: "Pt",
      atomicNumber: 78,
      mcxUnit: "10g",
      globalUnit: "troy oz",
      mcxSpotInr: 39820,
      globalSpotUsd: 1263.62,
      baseLandedInr: 37453,
      taxPaidParityInr: 40891,
      paritySpreadInr: -1071,
      paritySpreadPct: -2.62,
      dayChangePct: -1.00,
      dutyPct: 6.0,
      gstPct: 3.0,
      accuracy: "71.6%",
      r2: "0.849",
      maeInr: 490,
      rmseInr: 675,
      maeUsd: 15.40,
      rmseUsd: 21.20,
      sentimentScore: "-0.12 (Neutral / Soft)",
      topDriver: "South Africa Bushveld Mine Supply (33%) & Heavy-Duty/Hybrid Autocatalysts (26%)",
      forecasts: { "1D": "-0.24% (₹39,724)", "1W": "+0.65% (₹40,079)", "1M": "+2.15% (₹40,676)", "3M": "+4.90% (₹41,771)", "6M": "+8.30% (₹43,125)", "1Y": "+13.20% (₹45,076)", "5Y": "+44.80% (₹57,659)" }
    },
    copper: {
      name: "Copper",
      symbol: "Cu",
      atomicNumber: 29,
      mcxUnit: "1kg",
      globalUnit: "MT",
      mcxSpotInr: 984.50,
      globalSpotUsd: 9890.00,
      baseLandedInr: 911.76,
      taxPaidParityInr: 995.46,
      paritySpreadInr: -10.96,
      paritySpreadPct: -1.10,
      dayChangePct: 2.18,
      dutyPct: 5.0,
      gstPct: 18.0,
      accuracy: "79.1%",
      r2: "0.921",
      maeInr: 9.40,
      rmseInr: 12.80,
      maeUsd: 94.00,
      rmseUsd: 128.00,
      sentimentScore: "+0.49 (Bullish)",
      topDriver: "HVDC Power Grid & AI Data Centers (35%) & Chile/Peru Concentrate TC/RCs (24%)",
      forecasts: { "1D": "+0.78% (₹992.18)", "1W": "+2.15% (₹1,005.67)", "1M": "+5.40% (₹1,037.66)", "3M": "+9.60% (₹1,079.01)", "6M": "+14.80% (₹1,130.21)", "1Y": "+21.60% (₹1,197.15)", "5Y": "+76.40% (₹1,736.66)" }
    },
    aluminum: {
      name: "Aluminium",
      symbol: "Al",
      atomicNumber: 13,
      mcxUnit: "1kg",
      globalUnit: "MT",
      mcxSpotInr: 264.80,
      globalSpotUsd: 2685.00,
      baseLandedInr: 247.53,
      taxPaidParityInr: 270.25,
      paritySpreadInr: -5.45,
      paritySpreadPct: -2.02,
      dayChangePct: 1.42,
      dutyPct: 7.5,
      gstPct: 18.0,
      accuracy: "76.2%",
      r2: "0.895",
      maeInr: 3.10,
      rmseInr: 4.25,
      maeUsd: 31.00,
      rmseUsd: 43.00,
      sentimentScore: "+0.28 (Bullish)",
      topDriver: "Smelter Hydro-Power Caps & Alumina Feedstock (32%) & LME On-Warrant Stocks (25%)",
      forecasts: { "1D": "+0.45% (₹265.99)", "1W": "+1.35% (₹268.37)", "1M": "+3.60% (₹274.33)", "3M": "+6.40% (₹281.75)", "6M": "+9.80% (₹290.75)", "1Y": "+14.50% (₹303.20)", "5Y": "+48.00% (₹391.90)" }
    }
  }
};

app.get("/api/status", (_req, res) => {
  res.json({
    project: "Metal Prediction",
    metals: ["Gold", "Silver", "Platinum", "Copper", "Aluminium"],
    model: "MetalNLP-v1",
    mode: "local-trained-ml",
    nlpModel: metalChatbot.getMetadata()
  });
});

app.post("/api/chat", (req, res) => {
  const { message, sessionId, context } = req.body || {};

  if (
    typeof message !== "string" ||
    !message.trim() ||
    message.length > 2000 ||
    typeof sessionId !== "string" ||
    !/^[a-f0-9-]{36}$/i.test(sessionId)
  ) {
    return res.status(400).json({ error: "Invalid message or session." });
  }

  const result = metalChatbot.reply(message, sessionId, context || {}, METAL_PREDICT_KNOWLEDGE);
  return res.json(result);
});

// Health Check Endpoint for API-Offline Detection
app.get("/api/health", (_req, res) => {
  res.json({
    status: "ok",
    mode: "local-trained-ml",
    model: "MetalNLP-v1",
    nlpModel: metalChatbot.getMetadata(),
    asOfDate: METAL_PREDICT_KNOWLEDGE.asOfDate,
    timestamp: new Date().toISOString()
  });
});

// Dedicated Contact & Feedback Endpoint
app.post("/api/contact", (req, res) => {
  const name = typeof req.body?.name === "string" ? req.body.name.trim() : "";
  const email = typeof req.body?.email === "string" ? req.body.email.trim() : "";
  const category = typeof req.body?.category === "string" ? req.body.category.trim() : "General";
  const metal = typeof req.body?.metal === "string" ? req.body.metal.trim() : "All 5 Metals";
  const message = typeof req.body?.message === "string" ? req.body.message.trim() : "";

  if (!name || name.length > 100) {
    return res.status(400).json({ error: "Please provide a valid name (1–100 characters)." });
  }
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email) || email.length > 160) {
    return res.status(400).json({ error: "Please provide a valid email address." });
  }
  if (message.length < 10 || message.length > 2000) {
    return res.status(400).json({ error: "Message must be between 10 and 2000 characters." });
  }

  const ticketId = `MP-${Date.now().toString(36).toUpperCase()}`;
  const receivedAt = new Date().toISOString();
  feedbackSubmissions.push({ ticketId, name, email, category, metal, message, receivedAt });
  if (feedbackSubmissions.length > 200) feedbackSubmissions.shift();

  return res.json({ ok: true, ticketId, receivedAt });
});

// ============================================================================
// ZERO-COST SERVER-SIDE TTL CACHE FOR /api/prices (PROTECTS FREE API QUOTAS)
// ============================================================================
const PRICE_CACHE_TTL_MS = (Number(process.env.PRICE_CACHE_TTL_SECONDS) || 900) * 1000;
let marketPriceCache = { data: null, expiresAt: 0 };

async function buildMarketSnapshot() {
  const snapshot = {
    asOfDate: trainedCommodityModels?.metadata?.asOfDate || METAL_PREDICT_KNOWLEDGE.asOfDate,
    usdInrSpot: trainedCommodityModels?.metadata?.usdinrSpot || METAL_PREDICT_KNOWLEDGE.usdInrSpot,
    troyOzGrams: METAL_PREDICT_KNOWLEDGE.troyOzGrams,
    source: trainedCommodityModels ? "trained-ml-pipeline" : "quant-snapshot-cache",
    metals: {}
  };

  for (const [key, m] of Object.entries(METAL_PREDICT_KNOWLEDGE.metals)) {
    snapshot.metals[key] = { ...m };
    if (trainedCommodityModels?.forecast_snapshot?.[key]) {
      const liveM = trainedCommodityModels.forecast_snapshot[key];
      snapshot.metals[key].globalSpotUsd = liveM.globalSpotUsd;
      snapshot.metals[key].mcxSpotInr = liveM.mcxSpotInr;
      snapshot.metals[key].baseLandedInr = liveM.baseLandedInr;
      snapshot.metals[key].taxPaidParityInr = liveM.taxPaidParityInr;
      snapshot.metals[key].paritySpreadPct = liveM.paritySpreadPct;
      snapshot.metals[key].dayChangePct = liveM.dayChangePct;
      snapshot.metals[key].accuracy = liveM.accuracy;
      snapshot.metals[key].r2 = liveM.r2;
      snapshot.metals[key].maeInr = liveM.maeInr;
      snapshot.metals[key].rmseInr = liveM.rmseInr;
      snapshot.metals[key].forecasts = liveM.forecasts;
    }
  }

  const metalApiKey = process.env.METALPRICE_API_KEY ? process.env.METALPRICE_API_KEY.trim() : "";
  const hasLivePriceKey = Boolean(metalApiKey && metalApiKey !== "YOUR_METALPRICE_API_KEY");

  if (hasLivePriceKey) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 3500);
      const url = `https://api.metalpriceapi.com/v1/latest?api_key=${encodeURIComponent(metalApiKey)}&base=USD&currencies=INR,XAU,XAG,XPT`;
      const upstream = await fetch(url, { signal: controller.signal });
      clearTimeout(timeout);

      if (upstream.ok) {
        const payload = await upstream.json();
        const rates = payload && payload.rates ? payload.rates : null;
        if (rates && Number.isFinite(rates.INR) && rates.INR >= 70 && rates.INR <= 130) {
          snapshot.usdInrSpot = Number(rates.INR.toFixed(2));
        }

        const preciousMap = [
          { key: "gold", code: "USDXAU", grams: 10, minUsd: 1500, maxUsd: 8000, roundInt: true },
          { key: "silver", code: "USDXAG", grams: 1000, minUsd: 15, maxUsd: 150, roundInt: true },
          { key: "platinum", code: "USDXPT", grams: 10, minUsd: 500, maxUsd: 3500, roundInt: true }
        ];

        let updatedAny = false;
        for (const item of preciousMap) {
          const usdVal = rates && Number.isFinite(rates[item.code]) ? rates[item.code] : null;
          if (usdVal && usdVal >= item.minUsd && usdVal <= item.maxUsd) {
            const m = snapshot.metals[item.key];
            const prevParity = m.taxPaidParityInr || m.baseLandedInr * 1.0918;
            const prevPremiumRatio = prevParity > 0 ? m.mcxSpotInr / prevParity : 1;
            const baseLanded = (usdVal / METAL_PREDICT_KNOWLEDGE.troyOzGrams) * item.grams * snapshot.usdInrSpot;
            const taxPaid = baseLanded * 1.0918;
            const mcxSpot = taxPaid * prevPremiumRatio;
            m.globalSpotUsd = Number(usdVal.toFixed(2));
            m.baseLandedInr = Math.round(baseLanded);
            m.taxPaidParityInr = Math.round(taxPaid);
            m.mcxSpotInr = Math.round(mcxSpot);
            m.paritySpreadInr = Math.round(m.mcxSpotInr - m.taxPaidParityInr);
            m.paritySpreadPct = Number(((m.paritySpreadInr / m.taxPaidParityInr) * 100).toFixed(2));
            updatedAny = true;
          }
        }
        if (updatedAny) snapshot.source = "metalpriceapi-cached";
      }
    } catch (_err) {
      // Keep deterministic snapshot cache on timeout/upstream failure
    }
  }

  const now = Date.now();
  snapshot.cachedAt = new Date(now).toISOString();
  snapshot.expiresAt = new Date(now + PRICE_CACHE_TTL_MS).toISOString();
  snapshot.ttlSeconds = Math.round(PRICE_CACHE_TTL_MS / 1000);
  return snapshot;
}

// Server-Side TTL-Cached 5-Metal Spot & Parity Endpoint
app.get("/api/prices", async (_req, res) => {
  try {
    if (marketPriceCache.data && Date.now() < marketPriceCache.expiresAt) {
      return res.json({ ...marketPriceCache.data, cacheHit: true });
    }
    const fresh = await buildMarketSnapshot();
    marketPriceCache = {
      data: fresh,
      expiresAt: Date.now() + PRICE_CACHE_TTL_MS
    };
    return res.json({ ...fresh, cacheHit: false });
  } catch (_err) {
    return res.status(500).json({ error: "Unable to build cached market snapshot." });
  }
});

// Clean Legal & Compliance Routes
["/legal", "/disclaimer", "/privacy", "/terms", "/cookies"].forEach((routePath) => {
  app.get(routePath, (_req, res) => {
    res.sendFile(path.join(__dirname, "public", "legal.html"));
  });
});

app.get("/methodology", (_req, res) => {
  res.sendFile(path.join(__dirname, "public", "methodology.html"));
});

// Express 5 Catch-All 404 Handler
app.use((req, res) => {
  if (req.path.startsWith("/api/")) {
    return res.status(404).json({ error: `API endpoint ${req.method} ${req.path} not found.` });
  }
  return res.status(404).sendFile(path.join(__dirname, "public", "404.html"));
});

if (!process.env.VERCEL) {
  const PORT = Number(process.env.PORT) || 3000;
  app.listen(PORT, () => {
    const meta = metalChatbot.getMetadata();
    console.log(`Metal Prediction Server running at http://localhost:${PORT}`);
    console.log(`MetalNLP-v1 Custom Trained Chatbot Ready (Val Accuracy: ${meta.valAccuracy}%, Vocab: ${meta.vocabSize})`);
  });

  if (PORT !== 8765 && !process.env.RENDER && !process.env.RAILWAY_ENVIRONMENT) {
    app.listen(8765, () => {
      console.log("Metal Prediction Server also mirrored at http://localhost:8765");
    }).on("error", () => {});
  }
}

export default app;

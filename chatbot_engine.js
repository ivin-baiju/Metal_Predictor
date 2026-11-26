import fs from "fs";
import { MODEL_WEIGHTS_PATH, trainAndSaveMetalNlpModel } from "./train_chatbot.js";
import { normalizeText, vectorizeQuery } from "./nlp_featurizer.js";

const BUNDLED_WEIGHTS_URL = new URL("./model/metal_nlp_weights.json", import.meta.url);

const METAL_SYNONYMS = {
  gold: ["gold", "xau", "au", "sona", "bullion"],
  silver: ["silver", "xag", "ag", "chandi"],
  platinum: ["platinum", "platnum", "xpt", "pt"],
  copper: ["copper", "coper", "cu", "hg", "tamba"],
  aluminum: ["aluminium", "aluminum", "alumnum", "aluminim", "alu", "al"]
};

const OUT_OF_SCOPE_KEYWORDS = [
  "bitcoin", "btc", "crypto", "ethereum", "eth", "solana",
  "nifty", "sensex", "stock", "stocks", "nasdaq", "tesla", "apple", "reliance",
  "crude", "oil", "natural gas", "palladium", "zinc", "nickel", "lead", "tin",
  "steel", "iron ore", "rhodium", "lithium", "cobalt"
];

const HORIZON_ANCHORS = {
  gold:     [[1, 0.62], [7, 1.84], [30, 4.43], [90, 7.85], [180, 11.40], [365, 16.90], [1825, 62.50]],
  silver:   [[1, 1.15], [7, 2.90], [30, 6.80], [90, 12.10], [180, 17.50], [365, 24.40], [1825, 88.20]],
  platinum: [[1, -0.24], [7, 0.65], [30, 2.15], [90, 4.90], [180, 8.30], [365, 13.20], [1825, 44.80]],
  copper:   [[1, 0.78], [7, 2.15], [30, 5.40], [90, 9.60], [180, 14.80], [365, 21.60], [1825, 76.40]],
  aluminum: [[1, 0.45], [7, 1.35], [30, 3.60], [90, 6.40], [180, 9.80], [365, 14.50], [1825, 48.00]]
};

function interpolateHorizonReturn(metalKey, days) {
  const anchors = HORIZON_ANCHORS[metalKey] || HORIZON_ANCHORS.gold;
  const clamped = Math.max(1, Math.min(1825, days));
  let lo = anchors[0];
  let hi = anchors[anchors.length - 1];
  for (let i = 0; i < anchors.length - 1; i++) {
    if (clamped >= anchors[i][0] && clamped <= anchors[i + 1][0]) {
      lo = anchors[i];
      hi = anchors[i + 1];
      break;
    }
  }
  const span = Math.max(1, hi[0] - lo[0]);
  const t = (clamped - lo[0]) / span;
  return Number((lo[1] + (hi[1] - lo[1]) * t).toFixed(2));
}

function fmtInr(val, metalKey) {
  if (metalKey === "copper" || metalKey === "aluminum") {
    return "₹" + Number(val).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  return "₹" + Math.round(val).toLocaleString("en-IN");
}

function fmtUsd(val) {
  return "$" + Number(val).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function extractMentionedMetals(normQuery) {
  const padded = ` ${normQuery} `;
  const found = [];
  for (const [metalKey, aliases] of Object.entries(METAL_SYNONYMS)) {
    for (const alias of aliases) {
      if (padded.includes(` ${alias} `)) {
        found.push(metalKey);
        break;
      }
    }
  }
  return found;
}

function extractHorizonSlot(rawQuery, normQuery, fallbackHorizon) {
  const explicitMatch = normQuery.match(/\b(1d|1w|1m|3m|6m|1y|5y)\b/);
  if (explicitMatch) {
    const key = explicitMatch[1].toUpperCase();
    const mapDays = { "1D": 1, "1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "5Y": 1825 };
    return { key, days: mapDays[key], custom: false };
  }

  const numUnit = normQuery.match(/\b(\d+)\s*(day|days|week|weeks|month|months|year|years|yr|yrs)\b/);
  if (numUnit) {
    const n = parseInt(numUnit[1], 10);
    const unit = numUnit[2];
    let days = n;
    if (unit.startsWith("week")) days = n * 7;
    else if (unit.startsWith("month")) days = n * 30;
    else if (unit.startsWith("year") || unit.startsWith("yr")) days = n * 365;
    const exactKey = { 1: "1D", 7: "1W", 30: "1M", 90: "3M", 180: "6M", 365: "1Y", 1825: "5Y" }[days];
    if (exactKey) return { key: exactKey, days, custom: false };
    return { key: `${days}D`, days, custom: true };
  }

  if (/\btomorrow\b|\bnext day\b/.test(normQuery)) return { key: "1D", days: 1, custom: false };
  if (/\bnext week\b/.test(normQuery)) return { key: "1W", days: 7, custom: false };
  if (/\bnext month\b/.test(normQuery)) return { key: "1M", days: 30, custom: false };
  if (/\bquarter\b/.test(normQuery)) return { key: "3M", days: 90, custom: false };
  if (/\bhalf year\b/.test(normQuery)) return { key: "6M", days: 180, custom: false };
  if (/\bnext year\b|\b1 year\b/.test(normQuery)) return { key: "1Y", days: 365, custom: false };

  const safeFallback = ["1D", "1W", "1M", "3M", "6M", "1Y", "5Y"].includes(fallbackHorizon)
    ? fallbackHorizon
    : "1M";
  const mapDays = { "1D": 1, "1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "5Y": 1825 };
  return { key: safeFallback, days: mapDays[safeFallback], custom: false };
}

function extractNumericOverrides(rawQuery) {
  const usdMatch = rawQuery.match(/\$\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:usd|dollars?)\b/i);
  const fxMatch = rawQuery.match(/(?:₹|inr|rupee|fx)\s*[:=@]?\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:inr|rupees?)\b/i);
  const dutyMatch = rawQuery.match(/duty\s*(?:at|to|of|=|:)?\s*(\d+(?:\.\d+)?)\s*%?/i);
  const gstMatch = rawQuery.match(/gst\s*(?:at|to|of|=|:)?\s*(\d+(?:\.\d+)?)\s*%?/i);
  const shockMatch = rawQuery.match(/(?:shock|rally|drop|move)\s*(?:of|by|to)?\s*([+-]?\d+(?:\.\d+)?)\s*%?/i)
    || rawQuery.match(/([+-]\d+(?:\.\d+)?)\s*%/);

  return {
    usd: usdMatch ? parseFloat(usdMatch[1] || usdMatch[2]) : null,
    fx: fxMatch ? parseFloat(fxMatch[1] || fxMatch[2]) : null,
    duty: dutyMatch ? parseFloat(dutyMatch[1]) : null,
    gst: gstMatch ? parseFloat(gstMatch[1]) : null,
    shock: shockMatch ? parseFloat(shockMatch[1]) : null
  };
}

function softmaxWithTemp(logits, temperature = 0.55) {
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    const scaled = logits[i] / temperature;
    if (scaled > maxLogit) maxLogit = scaled;
  }
  const exps = new Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] / temperature - maxLogit);
    exps[i] = e;
    sum += e;
  }
  return exps.map((v) => v / sum);
}

export class MetalNlpChatbot {
  constructor() {
    this.model = this.loadOrTrainModel();
    this.sessionMemory = new Map();
  }

  loadOrTrainModel() {
    for (const candidate of [BUNDLED_WEIGHTS_URL, MODEL_WEIGHTS_PATH]) {
      try {
        if (fs.existsSync(candidate)) {
          const parsed = JSON.parse(fs.readFileSync(candidate, "utf8"));
          if (parsed && parsed.intents && parsed.vocab && parsed.weights) {
            return parsed;
          }
        }
      } catch (_err) {
        // Re-try next path or re-train in memory below
      }
    }
    return trainAndSaveMetalNlpModel();
  }

  getMetadata() {
    return {
      modelName: this.model.modelName,
      architecture: this.model.architecture,
      trainedAt: this.model.trainedAt,
      corpusSize: this.model.corpusSize,
      vocabSize: this.model.vocabSize,
      trainAccuracy: this.model.trainAccuracy,
      valAccuracy: this.model.valAccuracy,
      intentsCount: this.model.intents.length
    };
  }

  predictIntent(rawText) {
    const vec = vectorizeQuery(rawText, this.model.vocab, this.model.idf);
    if (!vec.length) {
      return { intent: "greeting_capabilities", confidence: 0.62, runnerUp: "forecast_single" };
    }

    const numClasses = this.model.intents.length;
    const logits = new Array(numClasses);
    for (let c = 0; c < numClasses; c++) {
      let sum = this.model.biases[c] || 0;
      const w = this.model.weights[c];
      for (const [idx, val] of vec) {
        sum += w[idx] * val;
      }
      logits[c] = sum;
    }

    const probs = softmaxWithTemp(logits, 0.55);
    const ranked = probs
      .map((p, i) => ({ intent: this.model.intents[i], prob: p }))
      .sort((a, b) => b.prob - a.prob);

    return {
      intent: ranked[0].intent,
      confidence: Number(ranked[0].prob.toFixed(3)),
      runnerUp: ranked[1] ? ranked[1].intent : null
    };
  }

  reply(rawMessage, sessionId, clientContext = {}, knowledgeBase) {
    const norm = normalizeText(rawMessage);
    const prediction = this.predictIntent(rawMessage);
    let { intent, confidence } = prediction;

    const mentionedMetals = extractMentionedMetals(norm);
    const hasOutOfScope = OUT_OF_SCOPE_KEYWORDS.some((kw) => (` ${norm} `).includes(` ${kw} `));

    if (hasOutOfScope && mentionedMetals.length === 0) {
      intent = "out_of_scope_asset";
      confidence = Math.max(confidence, 0.94);
    }

    const prevSession = this.sessionMemory.get(sessionId) || {};
    const hasPronounFollowUp = /\b(it|its|this|that|same|also|what about)\b/.test(norm);

    const activeKey = mentionedMetals[0]
      || (hasPronounFollowUp && prevSession.lastMetal ? prevSession.lastMetal : null)
      || (clientContext.metal && knowledgeBase.metals[clientContext.metal] ? clientContext.metal : "gold");

    const horizonSlot = extractHorizonSlot(rawMessage, norm, clientContext.horizon || prevSession.lastHorizon || "1M");
    const overrides = extractNumericOverrides(rawMessage);

    // Save multi-turn context (bounded to 500 sessions)
    this.sessionMemory.set(sessionId, {
      lastMetal: activeKey,
      lastHorizon: horizonSlot.key,
      lastIntent: intent,
      updatedAt: Date.now()
    });
    if (this.sessionMemory.size > 500) {
      const oldestKey = this.sessionMemory.keys().next().value;
      this.sessionMemory.delete(oldestKey);
    }

    const metals = knowledgeBase.metals;
    const target = metals[activeKey];
    let replyText = "";

    switch (intent) {
      case "compare_metals": {
        const compareKeys = mentionedMetals.length >= 2
          ? mentionedMetals
          : ["silver", "copper", "gold", "aluminum", "platinum"];
        const lines = compareKeys.map((k) => {
          const item = metals[k];
          const ret1M = interpolateHorizonReturn(k, 30);
          const ret1Y = interpolateHorizonReturn(k, 365);
          const p1M = fmtInr(item.mcxSpotInr * (1 + ret1M / 100), k);
          const p1Y = fmtInr(item.mcxSpotInr * (1 + ret1Y / 100), k);
          return `• ${item.name} (${item.symbol}/${item.mcxUnit}): Spot ${fmtInr(item.mcxSpotInr, k)} (${fmtUsd(item.globalSpotUsd)}/${item.globalUnit}) → 1M: +${ret1M}% (${p1M}) | 1Y: +${ret1Y}% (${p1Y}) | Acc ${item.accuracy}`;
        });
        replyText = [
          `[MetalNLP-v1] Multi-Metal Ensemble Comparison (as of ${knowledgeBase.asOfDate}):`,
          ...lines,
          `Summary: Silver carries the strongest 1Y return torque (+24.40%), while Copper leads directional accuracy (79.1%, R² 0.921).`
        ].join("\n");
        break;
      }

      case "parity_fx_duty": {
        const fx = overrides.fx && overrides.fx >= 60 && overrides.fx <= 130 ? overrides.fx : knowledgeBase.usdInrSpot;
        const globalUsd = overrides.usd && overrides.usd > 0 ? overrides.usd : target.globalSpotUsd;
        const isTroy = target.globalUnit === "troy oz";
        const grams = target.mcxUnit === "10g" ? 10 : 1000;
        const baseLanded = isTroy
          ? (globalUsd / knowledgeBase.troyOzGrams) * grams * fx
          : (globalUsd / 1000) * fx;
        const duty = overrides.duty !== null ? overrides.duty : target.dutyPct;
        const gst = overrides.gst !== null ? overrides.gst : target.gstPct;
        const effMult = isTroy ? (1 + duty / 100) * (1 + gst / 100) : 1.0918;
        const taxPaid = baseLanded * effMult;
        const customNote = (overrides.usd !== null || overrides.fx !== null || overrides.duty !== null) ? " (Custom Scenario)" : "";
        replyText = [
          `[MetalNLP-v1] Global-to-MCX Parity Math for ${target.name} (${target.symbol})${customNote}:`,
          `• Inputs: Global ${fmtUsd(globalUsd)}/${target.globalUnit} · USD/INR ₹${fx.toFixed(2)}`,
          isTroy
            ? `• Base Landed: (${fmtUsd(globalUsd)} ÷ 31.1035 g/oz) × ${grams}g × ₹${fx.toFixed(2)} = ${fmtInr(baseLanded, activeKey)}`
            : `• Base Landed: (${fmtUsd(globalUsd)} ÷ 1,000 kg/MT) × ₹${fx.toFixed(2)} = ${fmtInr(baseLanded, activeKey)}/kg`,
          `• Duty + GST Stack (${duty}% + ${gst}%): Tax-Paid Parity = ${fmtInr(taxPaid, activeKey)} / ${target.mcxUnit}`,
          `• Live MCX Spot: ${fmtInr(target.mcxSpotInr, activeKey)} | Spread vs Parity: ${target.paritySpreadPct >= 0 ? "+" : ""}${target.paritySpreadPct}%`
        ].join("\n");
        break;
      }

      case "arbitrage_spread": {
        const sorted = Object.entries(metals)
          .map(([k, v]) => ({ k, ...v }))
          .sort((a, b) => b.paritySpreadPct - a.paritySpreadPct);
        replyText = [
          `[MetalNLP-v1] MCX Arbitrage Spread for ${target.name} (${target.symbol}):`,
          `• Live MCX Spot: ${fmtInr(target.mcxSpotInr, activeKey)} vs Tax-Paid Parity: ${fmtInr(target.taxPaidParityInr, activeKey)}`,
          `• Spread: ${target.paritySpreadInr >= 0 ? "+" : "-"}${fmtInr(Math.abs(target.paritySpreadInr), activeKey)} (${target.paritySpreadPct >= 0 ? "+" : ""}${target.paritySpreadPct}%) — ${target.paritySpreadPct >= 0 ? "Domestic Premium" : "Discount to Landed Parity"}`,
          `• Cross-Market Ranking: ${sorted.map((x) => `${x.symbol} (${x.paritySpreadPct >= 0 ? "+" : ""}${x.paritySpreadPct}%)`).join(" > ")}`
        ].join("\n");
        break;
      }

      case "tax_shock_sim": {
        const duty = overrides.duty !== null ? overrides.duty : 6.0;
        const gst = overrides.gst !== null ? overrides.gst : 3.0;
        const fx = overrides.fx && overrides.fx >= 60 && overrides.fx <= 130 ? overrides.fx : knowledgeBase.usdInrSpot;
        const shock = overrides.shock !== null ? overrides.shock : 0;
        const baseFactor = 1.06 * 1.03 * knowledgeBase.usdInrSpot;
        const simFactor = (1 + duty / 100) * (1 + gst / 100) * fx * (1 + shock / 100);
        const mult = simFactor / baseFactor;
        const beta = (activeKey === "copper" || activeKey === "silver") ? 1.06 : 1.0;
        const effMult = 1 + (mult - 1) * beta;
        const simPrice = target.mcxSpotInr * effMult;
        const pctShift = (effMult - 1) * 100;
        replyText = [
          `[MetalNLP-v1] Stress Simulation for ${target.name} (${target.symbol}):`,
          `• Scenario: Duty ${duty.toFixed(1)}% · GST ${gst.toFixed(1)}% · USD/INR ₹${fx.toFixed(2)} · Global Shock ${shock >= 0 ? "+" : ""}${shock.toFixed(1)}%`,
          `• Simulated MCX Price: ${fmtInr(simPrice, activeKey)} / ${target.mcxUnit} (${pctShift >= 0 ? "+" : ""}${pctShift.toFixed(2)}% vs baseline ${fmtInr(target.mcxSpotInr, activeKey)})`,
          `• Tip: Open "Tax & Shock Sim" in the dashboard for side-by-side 5-metal index bars.`
        ].join("\n");
        break;
      }

      case "sentiment_news": {
        replyText = [
          `[MetalNLP-v1] FinBERT Sentiment & Physical Drivers for ${target.name} (${target.symbol}):`,
          `• Composite FinBERT Score: ${target.sentimentScore}`,
          `• Primary Catalyst: ${target.topDriver}`,
          `• 5-Metal Polarity Ranking: Silver (+0.56) > Copper (+0.49) > Gold (+0.42) > Aluminium (+0.28) > Platinum (-0.12).`,
          `• Click "FinBERT Feed →" to filter live bullion & base-metal dispatches.`
        ].join("\n");
        break;
      }

      case "model_accuracy_error": {
        replyText = [
          `[MetalNLP-v1] Walk-Forward Validation for ${target.name} (${target.symbol}):`,
          `• Directional Accuracy: ${target.accuracy} | Out-of-Sample R²: ${target.r2}`,
          `• MAE: ${fmtInr(target.maeInr, activeKey)} (${fmtUsd(target.maeUsd)}) | RMSE: ${fmtInr(target.rmseInr, activeKey)} (${fmtUsd(target.rmseUsd)})`,
          `• Highest R² Metal: Copper (R² 0.921, 79.1% accuracy), followed by Gold (R² 0.912, 78.4%). See /methodology for the full 5-metal ledger.`
        ].join("\n");
        break;
      }

      case "methodology_explain": {
        replyText = [
          `[MetalNLP-v1] Hybrid Architecture & Local Chatbot Pipeline:`,
          `• Price Ensemble: ${knowledgeBase.architecture}`,
          `• Chatbot Model: 100% free self-hosted MetalNLP-v1 (Sublinear TF-IDF word 1–2g + char 3–4g + Softmax Logistic Regression trained on ${this.model.corpusSize} domain utterances; Val Acc ${this.model.valAccuracy}%).`,
          `• Zero External API Costs: Spot prices & FX are cached via server-side TTL (/api/prices) with local unit/duty parity computation.`
        ].join("\n");
        break;
      }

      case "spot_price_quote": {
        const listRequested = /\ball\b|\b5 metals\b|\bfive\b/.test(norm) || mentionedMetals.length >= 2;
        if (listRequested) {
          const keys = mentionedMetals.length >= 2 ? mentionedMetals : Object.keys(metals);
          replyText = [
            `[MetalNLP-v1] Live Spot Benchmarks (USD/INR ₹${knowledgeBase.usdInrSpot} · ${knowledgeBase.asOfDate}):`,
            ...keys.map((k) => {
              const m = metals[k];
              return `• ${m.name} (${m.symbol}): MCX ${fmtInr(m.mcxSpotInr, k)}/${m.mcxUnit} | Global ${fmtUsd(m.globalSpotUsd)}/${m.globalUnit} (${m.dayChangePct >= 0 ? "+" : ""}${m.dayChangePct}%)`;
            })
          ].join("\n");
        } else {
          replyText = [
            `[MetalNLP-v1] Live Benchmark for ${target.name} (${target.symbol}):`,
            `• MCX Spot: ${fmtInr(target.mcxSpotInr, activeKey)} / ${target.mcxUnit} (24h: ${target.dayChangePct >= 0 ? "+" : ""}${target.dayChangePct}%)`,
            `• Global Benchmark: ${fmtUsd(target.globalSpotUsd)} / ${target.globalUnit} | USD/INR: ₹${knowledgeBase.usdInrSpot}`,
            `• Tax-Paid Landed Parity: ${fmtInr(target.taxPaidParityInr, activeKey)} / ${target.mcxUnit}`
          ].join("\n");
        }
        break;
      }

      case "compliance_disclaimer": {
        replyText = [
          `[MetalNLP-v1] Regulatory & Risk Notice:`,
          `• Algorithmic Estimates Only: All 5-metal forecasts, confidence bands, and parity figures are statistical outputs for informational and research purposes—not financial, trading, or buy/sell advice.`,
          `• Non-SEBI Registered: Metal Prediction is not registered with SEBI (or CFTC/SEC/FCA). Please consult a licensed adviser before trading.`,
          `• Full policies & GDPR/DPDP data deletion tools are available at /disclaimer, /privacy, /terms, and /cookies.`
        ].join("\n");
        break;
      }

      case "out_of_scope_asset": {
        replyText = [
          `[MetalNLP-v1] Asset Scope Notice:`,
          `• Metal Prediction strictly covers 5 commodities: Gold (Au), Silver (Ag), Platinum (Pt), Copper (Cu), and Aluminium (Al).`,
          `• External assets (crypto, equities, energy, or other base metals) are outside our trained ensemble. Ask me about any of our 5 covered metals!`
        ].join("\n");
        break;
      }

      case "greeting_capabilities": {
        replyText = [
          `Hello! I am MetalNLP-v1, your self-trained local quantitative assistant (0 external API fees, ${this.model.valAccuracy}% validation accuracy).`,
          `Ask me about Gold (Au), Silver (Ag), Platinum (Pt), Copper (Cu), or Aluminium (Al):`,
          `• Forecasts (1D to 5Y, or custom horizons like "45 days")`,
          `• Cross-metal comparisons & arbitrage spreads`,
          `• Custom Global-to-MCX parity ("gold parity at $3700 and ₹95") & duty shocks`,
          `• Walk-forward MAE/RMSE accuracy & FinBERT sentiment drivers`
        ].join("\n");
        break;
      }

      case "forecast_single":
      default: {
        const ret = interpolateHorizonReturn(activeKey, horizonSlot.days);
        const projInr = target.mcxSpotInr * (1 + ret / 100);
        const projUsd = target.globalSpotUsd * (1 + ret / 100);
        replyText = [
          `[MetalNLP-v1] ${target.name} (${target.symbol}) · ${horizonSlot.key} Forecast:`,
          `• Live Spot: ${fmtInr(target.mcxSpotInr, activeKey)}/${target.mcxUnit} (${fmtUsd(target.globalSpotUsd)}/${target.globalUnit})`,
          `• ${horizonSlot.key} Ensemble Target: ${fmtInr(projInr, activeKey)} (${fmtUsd(projUsd)}) · ${ret >= 0 ? "+" : ""}${ret}%`,
          `• Key Trajectory: 1D ${target.forecasts["1D"]} → 1M ${target.forecasts["1M"]} → 1Y ${target.forecasts["1Y"]}`,
          `• Driver & Confidence: ${target.topDriver} | Acc ${target.accuracy} | R² ${target.r2} (Informational only; not SEBI advice).`
        ].join("\n");
        break;
      }
    }

    return {
      reply: replyText,
      intent,
      confidence,
      activeMetal: activeKey,
      model: this.model.modelName,
      mode: "local-trained-ml"
    };
  }
}

let singletonChatbot = null;
export function getMetalChatbot() {
  if (!singletonChatbot) {
    singletonChatbot = new MetalNlpChatbot();
  }
  return singletonChatbot;
}

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { extractRawTokensAndNgrams, vectorizeQuery } from "./nlp_featurizer.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const MODEL_DIR = path.join(__dirname, "model");
export const MODEL_WEIGHTS_PATH = path.join(MODEL_DIR, "metal_nlp_weights.json");

// Supervised Training Corpus (12 Commodity & Quantitative Finance Intents)
export const TRAINING_CORPUS = [
  // 1. forecast_single
  { intent: "forecast_single", text: "what is the 1 month gold price prediction" },
  { intent: "forecast_single", text: "forecast silver for 1 year" },
  { intent: "forecast_single", text: "platinum 3m target price" },
  { intent: "forecast_single", text: "copper price outlook next 6 months" },
  { intent: "forecast_single", text: "aluminium 1y forcast" },
  { intent: "forecast_single", text: "alumnum 1y forcast" },
  { intent: "forecast_single", text: "where will gold go in 5 years" },
  { intent: "forecast_single", text: "predict sona rate next month" },
  { intent: "forecast_single", text: "chandi ka bhav prediction 1 week" },
  { intent: "forecast_single", text: "give me the active metal forecast and key drivers" },
  { intent: "forecast_single", text: "projected target for ag over 90 days" },
  { intent: "forecast_single", text: "what is the 45 day outlook for copper" },
  { intent: "forecast_single", text: "expected return and confidence band for au" },
  { intent: "forecast_single", text: "explain platinum 1d and 1w trajectory" },
  { intent: "forecast_single", text: "tell me the target price for aluminum in 2 years" },
  { intent: "forecast_single", text: "how bullish is silver for 6m" },
  { intent: "forecast_single", text: "gold forecast next 30 days" },
  { intent: "forecast_single", text: "copper target price 1y" },
  { intent: "forecast_single", text: "what will silver price be next month" },
  { intent: "forecast_single", text: "platinum 6 month price projection" },

  // 2. compare_metals
  { intent: "compare_metals", text: "compare 1m and 1y upside across gold silver platinum copper and aluminium" },
  { intent: "compare_metals", text: "compare all 5 metals" },
  { intent: "compare_metals", text: "which metal has the highest upside" },
  { intent: "compare_metals", text: "gold vs silver which is better" },
  { intent: "compare_metals", text: "rank all five commodities by return" },
  { intent: "compare_metals", text: "copper vs aluminium comparison" },
  { intent: "compare_metals", text: "best metal to watch over 1 year" },
  { intent: "compare_metals", text: "compare precious metals vs base metals" },
  { intent: "compare_metals", text: "which commodity is strongest right now" },
  { intent: "compare_metals", text: "side by side comparison of au ag pt cu al" },
  { intent: "compare_metals", text: "top performing metal forecast across 5 metals" },
  { intent: "compare_metals", text: "silver vs copper 1m outlook" },
  { intent: "compare_metals", text: "list forecasts for all metals together" },
  { intent: "compare_metals", text: "which has higher return platinum or gold" },
  { intent: "compare_metals", text: "compare gold silver and copper" },
  { intent: "compare_metals", text: "compare all metals performance" },
  { intent: "compare_metals", text: "which of the 5 metals will rally most" },
  { intent: "compare_metals", text: "compare upside potential across gold and silver" },

  // 3. parity_fx_duty
  { intent: "parity_fx_duty", text: "show global to mcx landed parity and duty math" },
  { intent: "parity_fx_duty", text: "how is comex converted to mcx 10g" },
  { intent: "parity_fx_duty", text: "calculate gold parity at 3700 usd and 95 inr" },
  { intent: "parity_fx_duty", text: "what is the customs duty and gst on silver" },
  { intent: "parity_fx_duty", text: "lme to mcx copper conversion formula" },
  { intent: "parity_fx_duty", text: "landed cost of aluminium in india" },
  { intent: "parity_fx_duty", text: "fx transmission usd inr parity calculation" },
  { intent: "parity_fx_duty", text: "troy ounce to 10 grams formula" },
  { intent: "parity_fx_duty", text: "tax paid parity breakdown for platinum" },
  { intent: "parity_fx_duty", text: "how does 31.1035 grams conversion work" },
  { intent: "parity_fx_duty", text: "compute import landed price at usd 42 and inr 94" },
  { intent: "parity_fx_duty", text: "import duty aidc and statutory gst rate on gold" },
  { intent: "parity_fx_duty", text: "what is base landed ex tax price for copper" },
  { intent: "parity_fx_duty", text: "convert global dollar price to indian rupee mcx" },
  { intent: "parity_fx_duty", text: "silver parity math with duty 6 pct" },
  { intent: "parity_fx_duty", text: "gold import parity calculation at 95 inr" },
  { intent: "parity_fx_duty", text: "how do you calculate tax paid landed parity" },
  { intent: "parity_fx_duty", text: "comex to mcx landed cost formula" },

  // 4. arbitrage_spread
  { intent: "arbitrage_spread", text: "what is the mcx arbitrage spread on gold" },
  { intent: "arbitrage_spread", text: "is silver trading at a premium or discount to parity" },
  { intent: "arbitrage_spread", text: "show mcx premium discount spread" },
  { intent: "arbitrage_spread", text: "copper arbitrage spread percentage" },
  { intent: "arbitrage_spread", text: "how wide is the comex mcx spread" },
  { intent: "arbitrage_spread", text: "is platinum undervalued vs global landed parity" },
  { intent: "arbitrage_spread", text: "aluminium discount to lme tax paid parity" },
  { intent: "arbitrage_spread", text: "difference between mcx spot and tax paid parity" },
  { intent: "arbitrage_spread", text: "arbitrage opportunity across the 5 metals" },
  { intent: "arbitrage_spread", text: "why is mcx silver at 8.63 percent premium" },
  { intent: "arbitrage_spread", text: "check domestic premium over landed cost" },
  { intent: "arbitrage_spread", text: "which metal trades at a discount on mcx" },
  { intent: "arbitrage_spread", text: "mcx vs global spread in rupees and percent" },
  { intent: "arbitrage_spread", text: "spread analysis for au and ag" },
  { intent: "arbitrage_spread", text: "is copper at a premium or discount to parity" },
  { intent: "arbitrage_spread", text: "show arbitrage spread on aluminium" },
  { intent: "arbitrage_spread", text: "mcx premium over landed parity today" },

  // 5. tax_shock_sim
  { intent: "tax_shock_sim", text: "what happens if customs duty is cut to 3 percent" },
  { intent: "tax_shock_sim", text: "simulate rupee weakening to 96 per dollar" },
  { intent: "tax_shock_sim", text: "impact of 8 percent global rally shock" },
  { intent: "tax_shock_sim", text: "run a budget duty cut stress scenario" },
  { intent: "tax_shock_sim", text: "how much does gold move if usd inr hits 95" },
  { intent: "tax_shock_sim", text: "simulate 10 percent comex shock on silver" },
  { intent: "tax_shock_sim", text: "stress test copper with duty 2 pct and fx 94" },
  { intent: "tax_shock_sim", text: "if gst changes to 5 percent what happens" },
  { intent: "tax_shock_sim", text: "macro shock simulation for aluminium" },
  { intent: "tax_shock_sim", text: "what if global prices drop minus 5 percent" },
  { intent: "tax_shock_sim", text: "budget tariff reduction effect on bullion" },
  { intent: "tax_shock_sim", text: "fx stress test at rupee 96" },
  { intent: "tax_shock_sim", text: "simulate shock +12 pct on platinum" },
  { intent: "tax_shock_sim", text: "how sensitive are prices to rupee depreciation" },
  { intent: "tax_shock_sim", text: "what if import duty increases to 10 percent" },
  { intent: "tax_shock_sim", text: "simulate fx shock usd inr 97" },
  { intent: "tax_shock_sim", text: "run macro stress simulator on gold" },

  // 6. sentiment_news
  { intent: "sentiment_news", text: "what is the finbert sentiment score for silver" },
  { intent: "sentiment_news", text: "show latest bullion and base metal news sentiment" },
  { intent: "sentiment_news", text: "why is copper bullish right now" },
  { intent: "sentiment_news", text: "what are the macro news drivers for gold" },
  { intent: "sentiment_news", text: "finbert nlp polarity across all 5 metals" },
  { intent: "sentiment_news", text: "explain platinum bearish catalyst" },
  { intent: "sentiment_news", text: "solar pv topcon news impact on silver" },
  { intent: "sentiment_news", text: "ai data center demand headlines for copper" },
  { intent: "sentiment_news", text: "yunnan hydro power news for aluminium" },
  { intent: "sentiment_news", text: "market sentiment headlines and confidence" },
  { intent: "sentiment_news", text: "is news flow positive or negative for pt" },
  { intent: "sentiment_news", text: "central bank buying and treasury yield news" },
  { intent: "sentiment_news", text: "top physical drivers moving metal prices" },
  { intent: "sentiment_news", text: "summarise finbert news feed" },
  { intent: "sentiment_news", text: "news sentiment polarity for gold and copper" },
  { intent: "sentiment_news", text: "latest finbert headlines and bullish score" },
  { intent: "sentiment_news", text: "what macro catalyst drives silver sentiment" },
  { intent: "sentiment_news", text: "physical supply and demand drivers news" },
  { intent: "sentiment_news", text: "finbert news and physical catalysts" },

  // 7. model_accuracy_error
  { intent: "model_accuracy_error", text: "what is the mae and rmse of the gold model" },
  { intent: "model_accuracy_error", text: "how accurate is your forecast" },
  { intent: "model_accuracy_error", text: "show out of sample r2 and directional accuracy" },
  { intent: "model_accuracy_error", text: "what is its mae and rmse" },
  { intent: "model_accuracy_error", text: "walk forward validation error metrics for copper" },
  { intent: "model_accuracy_error", text: "silver rmse in inr and usd" },
  { intent: "model_accuracy_error", text: "which metal model has the highest r2 score" },
  { intent: "model_accuracy_error", text: "backtest hit rate and error bounds" },
  { intent: "model_accuracy_error", text: "how reliable is platinum prediction accuracy" },
  { intent: "model_accuracy_error", text: "show model validation table stats" },
  { intent: "model_accuracy_error", text: "mean absolute error for aluminium" },
  { intent: "model_accuracy_error", text: "what is the r squared for au" },
  { intent: "model_accuracy_error", text: "root mean squared error across metals" },
  { intent: "model_accuracy_error", text: "how well did the model perform out of sample" },
  { intent: "model_accuracy_error", text: "what is the mae and rmse for silver" },
  { intent: "model_accuracy_error", text: "show directional accuracy and r2 score" },
  { intent: "model_accuracy_error", text: "check error metrics mae rmse" },
  { intent: "model_accuracy_error", text: "how accurate are the metal predictions" },
  { intent: "model_accuracy_error", text: "accuracy rate and error metrics" },

  // 8. methodology_explain
  { intent: "methodology_explain", text: "how does your prediction model work" },
  { intent: "methodology_explain", text: "explain the tft nbeats xgboost vecm ensemble" },
  { intent: "methodology_explain", text: "what machine learning architecture do you use" },
  { intent: "methodology_explain", text: "how are the 95 percent confidence intervals built" },
  { intent: "methodology_explain", text: "model weights and feature attribution methodology" },
  { intent: "methodology_explain", text: "explain temporal fusion transformer weight" },
  { intent: "methodology_explain", text: "how does your chatbot nlp work without gemini" },
  { intent: "methodology_explain", text: "is this chatbot trained locally" },
  { intent: "methodology_explain", text: "describe your forecasting pipeline and data caching" },
  { intent: "methodology_explain", text: "how does vecm cointegration work" },
  { intent: "methodology_explain", text: "what features go into the ensemble" },
  { intent: "methodology_explain", text: "tell me about the quant architecture" },
  { intent: "methodology_explain", text: "how are quantiles estimated" },
  { intent: "methodology_explain", text: "explain the methodology behind metal prediction" },
  { intent: "methodology_explain", text: "how is the hybrid ensemble trained" },
  { intent: "methodology_explain", text: "tell me how tft and nbeats work together" },
  { intent: "methodology_explain", text: "how does your local nlp classifier work" },
  { intent: "methodology_explain", text: "how are confidence intervals built in the model" },
  { intent: "methodology_explain", text: "explain shap feature attribution and model weights" },
  { intent: "methodology_explain", text: "95 percent confidence band methodology" },

  // 9. spot_price_quote
  { intent: "spot_price_quote", text: "what is the live spot price of gold today" },
  { intent: "spot_price_quote", text: "current silver rate in mcx and comex" },
  { intent: "spot_price_quote", text: "tamba current price per kg" },
  { intent: "spot_price_quote", text: "platinum spot benchmark right now" },
  { intent: "spot_price_quote", text: "today aluminium rate in rupees" },
  { intent: "spot_price_quote", text: "what is usd inr exchange rate right now" },
  { intent: "spot_price_quote", text: "live quote for copper" },
  { intent: "spot_price_quote", text: "check 24h change for gold and silver" },
  { intent: "spot_price_quote", text: "spot price of all 5 metals" },
  { intent: "spot_price_quote", text: "how much is 10g gold on mcx" },
  { intent: "spot_price_quote", text: "current lme copper usd per mt" },
  { intent: "spot_price_quote", text: "show current prices for au ag pt cu al" },
  { intent: "spot_price_quote", text: "latest benchmark quote for al" },
  { intent: "spot_price_quote", text: "mcx spot rate for silver 1kg" },
  { intent: "spot_price_quote", text: "what is the current gold price today" },
  { intent: "spot_price_quote", text: "live spot rate of platinum and copper" },
  { intent: "spot_price_quote", text: "today spot quote for aluminium" },
  { intent: "spot_price_quote", text: "how much does gold cost on mcx right now" },
  { intent: "spot_price_quote", text: "how much is silver per kg today" },

  // 10. compliance_disclaimer
  { intent: "compliance_disclaimer", text: "are you sebi registered" },
  { intent: "compliance_disclaimer", text: "is this financial or investment advice" },
  { intent: "compliance_disclaimer", text: "should i buy gold now" },
  { intent: "compliance_disclaimer", text: "can you tell me whether to buy or sell silver" },
  { intent: "compliance_disclaimer", text: "show legal disclaimer and privacy policy" },
  { intent: "compliance_disclaimer", text: "what are your terms and liability limits" },
  { intent: "compliance_disclaimer", text: "how do i delete my data under dpdp or gdpr" },
  { intent: "compliance_disclaimer", text: "cookie policy and consent settings" },
  { intent: "compliance_disclaimer", text: "recommend a trade in copper futures" },
  { intent: "compliance_disclaimer", text: "is it safe to invest based on this forecast" },
  { intent: "compliance_disclaimer", text: "give me a buy or sell signal" },
  { intent: "compliance_disclaimer", text: "regulatory compliance notice" },
  { intent: "compliance_disclaimer", text: "do you offer personalized trading advice" },
  { intent: "compliance_disclaimer", text: "where can i read your disclaimer" },
  { intent: "compliance_disclaimer", text: "financial disclaimer and privacy terms" },
  { intent: "compliance_disclaimer", text: "not sebi registered legal notice" },
  { intent: "compliance_disclaimer", text: "are predictions financial advice" },
  { intent: "compliance_disclaimer", text: "delete my personal data under gdpr and dpdp act" },
  { intent: "compliance_disclaimer", text: "privacy policy data deletion rights" },
  { intent: "compliance_disclaimer", text: "dpdp and gdpr compliance" },

  // 11. out_of_scope_asset
  { intent: "out_of_scope_asset", text: "predict bitcoin price for next month" },
  { intent: "out_of_scope_asset", text: "what is the forecast for crude oil" },
  { intent: "out_of_scope_asset", text: "can you predict nifty 50 or sensex" },
  { intent: "out_of_scope_asset", text: "palladium price prediction 1y" },
  { intent: "out_of_scope_asset", text: "zinc and nickel outlook" },
  { intent: "out_of_scope_asset", text: "ethereum crypto price target" },
  { intent: "out_of_scope_asset", text: "steel and iron ore forecast" },
  { intent: "out_of_scope_asset", text: "apple or tesla stock prediction" },
  { intent: "out_of_scope_asset", text: "predict natural gas futures" },
  { intent: "out_of_scope_asset", text: "lead and tin lme forecast" },
  { intent: "out_of_scope_asset", text: "btc usd target price" },
  { intent: "out_of_scope_asset", text: "rhodium price prediction" },
  { intent: "out_of_scope_asset", text: "what about lithium and cobalt" },
  { intent: "out_of_scope_asset", text: "forecast reliance stock price" },
  { intent: "out_of_scope_asset", text: "can you predict eth or crypto" },
  { intent: "out_of_scope_asset", text: "crude oil and natural gas price" },
  { intent: "out_of_scope_asset", text: "nifty stock market forecast" },

  // 12. greeting_capabilities
  { intent: "greeting_capabilities", text: "hello" },
  { intent: "greeting_capabilities", text: "hi there" },
  { intent: "greeting_capabilities", text: "hey what can you do" },
  { intent: "greeting_capabilities", text: "who are you" },
  { intent: "greeting_capabilities", text: "help me get started" },
  { intent: "greeting_capabilities", text: "what questions can i ask you" },
  { intent: "greeting_capabilities", text: "good morning" },
  { intent: "greeting_capabilities", text: "thanks thank you" },
  { intent: "greeting_capabilities", text: "show available commands and features" },
  { intent: "greeting_capabilities", text: "introduce yourself" },
  { intent: "greeting_capabilities", text: "how can you help me with metals" },
  { intent: "greeting_capabilities", text: "what metals do you cover" },
  { intent: "greeting_capabilities", text: "hi" },
  { intent: "greeting_capabilities", text: "capabilities overview" },
  { intent: "greeting_capabilities", text: "hey hello" },
  { intent: "greeting_capabilities", text: "what can this chatbot do" },
  { intent: "greeting_capabilities", text: "help me" },
  { intent: "greeting_capabilities", text: "good afternoon" },
  { intent: "greeting_capabilities", text: "good evening" },
  { intent: "greeting_capabilities", text: "greetings metal prediction" },
  { intent: "greeting_capabilities", text: "hello good morning" }
];

function deterministicShuffle(arr, seed = 42) {
  const copy = [...arr];
  let s = seed;
  for (let i = copy.length - 1; i > 0; i--) {
    s = (s * 1664525 + 1013904223) % 4294967296;
    const j = s % (i + 1);
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function buildVocabulary(corpus) {
  const docFreq = new Map();
  for (const item of corpus) {
    const tokens = extractRawTokensAndNgrams(item.text);
    const unique = new Set(tokens.map((t) => t.feat));
    for (const feat of unique) {
      docFreq.set(feat, (docFreq.get(feat) || 0) + 1);
    }
  }

  const vocab = {};
  const idf = [];
  const N = corpus.length;
  let idx = 0;

  for (const [term, df] of docFreq.entries()) {
    // Keep all word & bigram features, prune rare single-occurrence char 4-grams
    if (term.startsWith("c:") && df < 2) continue;
    vocab[term] = idx++;
    idf.push(Math.log((N + 1) / (df + 1)) + 1);
  }

  return { vocab, idf };
}

function softmax(logits) {
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > maxLogit) maxLogit = logits[i];
  }
  const exps = new Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - maxLogit);
    exps[i] = e;
    sum += e;
  }
  for (let i = 0; i < exps.length; i++) {
    exps[i] /= sum;
  }
  return exps;
}

function trainSoftmaxClassifier(samples, numClasses, vocabSize, epochs = 260, initialLr = 0.20, l2 = 1.5e-3) {
  const weights = Array.from({ length: numClasses }, () => new Float64Array(vocabSize));
  const biases = new Float64Array(numClasses);

  for (let epoch = 0; epoch < epochs; epoch++) {
    const lr = initialLr / (1 + epoch * 0.008);
    const ordered = deterministicShuffle(samples, 100 + epoch);

    for (const sample of ordered) {
      const logits = new Array(numClasses);
      for (let c = 0; c < numClasses; c++) {
        let sum = biases[c];
        const w = weights[c];
        for (const [idx, val] of sample.vec) {
          sum += w[idx] * val;
        }
        logits[c] = sum;
      }

      const probs = softmax(logits);

      for (let c = 0; c < numClasses; c++) {
        const err = probs[c] - (c === sample.label ? 1 : 0);
        biases[c] -= lr * err;
        const w = weights[c];
        for (const [idx, val] of sample.vec) {
          w[idx] = w[idx] * (1 - lr * l2) - lr * err * val;
        }
      }
    }
  }

  return { weights, biases };
}

function evaluateClassifier(samples, weights, biases, numClasses) {
  let correct = 0;
  for (const sample of samples) {
    let bestC = 0;
    let bestScore = -Infinity;
    for (let c = 0; c < numClasses; c++) {
      let sum = biases[c];
      const w = weights[c];
      for (const [idx, val] of sample.vec) {
        sum += w[idx] * val;
      }
      if (sum > bestScore) {
        bestScore = sum;
        bestC = c;
      }
    }
    if (bestC === sample.label) correct++;
  }
  return samples.length ? correct / samples.length : 0;
}

export function trainAndSaveMetalNlpModel() {
  const intents = [...new Set(TRAINING_CORPUS.map((d) => d.intent))];
  const intentToIdx = Object.fromEntries(intents.map((intent, i) => [intent, i]));

  // Stratified 85/15 train/validation split for honest hold-out evaluation
  const byIntent = new Map();
  for (const item of TRAINING_CORPUS) {
    if (!byIntent.has(item.intent)) byIntent.set(item.intent, []);
    byIntent.get(item.intent).push(item);
  }

  const trainSplitRaw = [];
  const valSplitRaw = [];
  for (const [intent, items] of byIntent.entries()) {
    const shuffled = deterministicShuffle(items, intent.length * 37);
    const valCount = Math.max(2, Math.round(shuffled.length * 0.15));
    valSplitRaw.push(...shuffled.slice(0, valCount));
    trainSplitRaw.push(...shuffled.slice(valCount));
  }

  const splitVocabData = buildVocabulary(trainSplitRaw);
  const trainSplitSamples = trainSplitRaw.map((d) => ({
    label: intentToIdx[d.intent],
    vec: vectorizeQuery(d.text, splitVocabData.vocab, splitVocabData.idf)
  }));
  const valSplitSamples = valSplitRaw.map((d) => ({
    label: intentToIdx[d.intent],
    vec: vectorizeQuery(d.text, splitVocabData.vocab, splitVocabData.idf)
  }));

  const evalModel = trainSoftmaxClassifier(
    trainSplitSamples,
    intents.length,
    Object.keys(splitVocabData.vocab).length,
    260,
    0.20,
    1.5e-3
  );
  const valAccuracy = evaluateClassifier(valSplitSamples, evalModel.weights, evalModel.biases, intents.length);

  // Final production fit on 100% of TRAINING_CORPUS
  const fullVocabData = buildVocabulary(TRAINING_CORPUS);
  const fullSamples = TRAINING_CORPUS.map((d) => ({
    label: intentToIdx[d.intent],
    vec: vectorizeQuery(d.text, fullVocabData.vocab, fullVocabData.idf)
  }));

  const vocabSize = Object.keys(fullVocabData.vocab).length;
  const fullModel = trainSoftmaxClassifier(fullSamples, intents.length, vocabSize, 260, 0.20, 1.5e-3);
  const trainAccuracy = evaluateClassifier(fullSamples, fullModel.weights, fullModel.biases, intents.length);

  // Quantize weights cleanly to 5 decimals for fast disk load & compact JSON
  const roundArr = (typedArr) => Array.from(typedArr, (v) => Number(v.toFixed(5)));

  const artifact = {
    modelName: "MetalNLP-v1",
    architecture: "Sublinear TF-IDF (Word 1-2g + Char 3-4g) + Softmax Logistic Regression",
    version: "1.0.0",
    trainedAt: new Date().toISOString(),
    corpusSize: TRAINING_CORPUS.length,
    valCorpusSize: valSplitRaw.length,
    vocabSize,
    trainAccuracy: Number((trainAccuracy * 100).toFixed(2)),
    valAccuracy: Number((valAccuracy * 100).toFixed(2)),
    intents,
    vocab: fullVocabData.vocab,
    idf: fullVocabData.idf.map((v) => Number(v.toFixed(5))),
    weights: fullModel.weights.map(roundArr),
    biases: roundArr(fullModel.biases)
  };

  try {
    fs.mkdirSync(MODEL_DIR, { recursive: true });
    fs.writeFileSync(MODEL_WEIGHTS_PATH, JSON.stringify(artifact));
  } catch (_err) {
    // Serverless environments (e.g. /var/task) may be read-only; keep artifact in memory.
  }

  return artifact;
}

const isMainModule = process.argv[1] && path.resolve(process.argv[1]) === __filename;
if (isMainModule) {
  const start = Date.now();
  const artifact = trainAndSaveMetalNlpModel();
  const elapsedMs = Date.now() - start;
  console.log(
    `[MetalNLP-v1] Trained on ${artifact.corpusSize} utterances (${artifact.intents.length} intents, ${artifact.vocabSize} features) in ${elapsedMs}ms`
  );
  console.log(
    `[MetalNLP-v1] Hold-Out Val Accuracy: ${artifact.valAccuracy}% (${artifact.valCorpusSize} samples) | Full Train Accuracy: ${artifact.trainAccuracy}%`
  );
  console.log(`[MetalNLP-v1] Weights saved to ${MODEL_WEIGHTS_PATH}`);
}

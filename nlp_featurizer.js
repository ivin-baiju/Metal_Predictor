// Shared Entity-Masked TF-IDF + Stem + Word/Char N-Gram Featurizer for MetalNLP-v1
// Used identically during training (train_chatbot.js) and inference (chatbot_engine.js).

const COVERED_METAL_GROUPS = [
  /\b(gold|xau|au|sona|bullion)\b/g,
  /\b(silver|xag|ag|chandi)\b/g,
  /\b(platinum|platnum|xpt|pt)\b/g,
  /\b(copper|coper|cu|hg|tamba)\b/g,
  /\b(aluminium|aluminum|alumnum|aluminim|alu|al)\b/g
];

const OOS_ASSET_REGEX = /\b(bitcoin|btc|crypto|ethereum|eth|solana|nifty|sensex|stock|stocks|nasdaq|tesla|apple|reliance|crude|oil|gas|palladium|zinc|nickel|lead|tin|steel|iron|rhodium|lithium|cobalt)\b/g;
const HORIZON_PHRASE_REGEX = /\b(1d|1w|1m|3m|6m|1y|5y|\d+\s*(?:days?|weeks?|months?|years?|yrs?))\b/g;

const GRAMMAR_STOPWORDS = new Set([
  "a", "an", "the", "is", "are", "am", "was", "were", "be",
  "do", "does", "did", "i", "me", "my", "you", "your", "we", "our",
  "of", "in", "on", "at", "to", "for", "from", "with", "by", "as",
  "or", "and", "please", "tell", "show", "give", "check", "can", "could", "would"
]);

const CHAR_STOPWORDS = new Set([
  "what", "when", "where", "which", "about", "into", "that", "this", "have", "will",
  "metal_tok", "horizon_tok", "oos_asset_tok", "ci95_tok", "num_tok"
]);

export function normalizeText(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/₹/g, " inr ")
    .replace(/\$/g, " usd ")
    .replace(/%/g, " pct ")
    .replace(/r²/g, " r2 ")
    .replace(/[^a-z0-9.\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function maskDomainEntities(norm) {
  let metalGroupsMatched = 0;
  let masked = norm;

  for (const re of COVERED_METAL_GROUPS) {
    re.lastIndex = 0;
    if (re.test(masked)) {
      metalGroupsMatched++;
    }
    re.lastIndex = 0;
    masked = masked.replace(re, " metal_tok ");
  }

  masked = masked
    .replace(/\b95\s+(?:pct|percent)\s+confidence\b/g, " ci95_tok confidence ")
    .replace(OOS_ASSET_REGEX, " oos_asset_tok ")
    .replace(HORIZON_PHRASE_REGEX, " horizon_tok ")
    .replace(/\b\d+(?:\.\d+)?\b/g, " num_tok ")
    .replace(/\s+/g, " ")
    .trim();

  return { masked, metalGroupsMatched };
}

export function extractRawTokensAndNgrams(text) {
  const norm = normalizeText(text);
  if (!norm) return [];

  const { masked, metalGroupsMatched } = maskDomainEntities(norm);
  const rawWords = masked.split(" ").filter(Boolean);
  const filtered = rawWords.filter((w) => !GRAMMAR_STOPWORDS.has(w));
  const words = filtered.length > 0 ? filtered : rawWords;
  const features = [];

  if (metalGroupsMatched >= 2) {
    features.push({ feat: "w:multi_metal_compare", weight: 2.6 });
  }

  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    features.push({ feat: `w:${w}`, weight: 2.2 });

    if (i < words.length - 1) {
      features.push({ feat: `b:${w}_${words[i + 1]}`, weight: 1.7 });
    }

    // 5-char stem for morphological generalization (e.g. accurate/accuracy, delete/deletion)
    if (
      w.length >= 5 &&
      w !== "metal_tok" &&
      w !== "horizon_tok" &&
      w !== "oos_asset_tok" &&
      w !== "ci95_tok" &&
      w !== "num_tok"
    ) {
      features.push({ feat: `s:${w.slice(0, 5)}`, weight: 1.4 });
    }

    // Character 4-grams on meaningful non-masked words (>=4 chars) for typo resilience
    if (w.length >= 4 && !CHAR_STOPWORDS.has(w)) {
      const padded = `^${w}$`;
      for (let j = 0; j <= padded.length - 4; j++) {
        features.push({ feat: `c:${padded.slice(j, j + 4)}`, weight: 0.4 });
      }
    }
  }

  return features;
}

export function vectorizeQuery(text, vocabMap, idfArray) {
  const rawFeatures = extractRawTokensAndNgrams(text);
  const weightedCounts = new Map();

  for (const { feat, weight } of rawFeatures) {
    const idx = vocabMap[feat];
    if (idx !== undefined) {
      weightedCounts.set(idx, (weightedCounts.get(idx) || 0) + weight);
    }
  }

  const sparse = [];
  let normSq = 0;

  for (const [idx, wCount] of weightedCounts.entries()) {
    const tf = 1 + Math.log(1 + wCount);
    const val = tf * idfArray[idx];
    sparse.push([idx, val]);
    normSq += val * val;
  }

  if (normSq > 0) {
    const invNorm = 1 / Math.sqrt(normSq);
    for (let i = 0; i < sparse.length; i++) {
      sparse[i][1] *= invNorm;
    }
  }

  return sparse;
}

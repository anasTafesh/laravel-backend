from flask import Flask, request, jsonify
import os, glob
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from keybert import KeyBERT
from transformers import pipeline
from collections import Counter
from sklearn.exceptions import ConvergenceWarning
import warnings
from flask_cors import CORS
from datasets import Dataset
import re
from urllib.parse import urlparse
import praw
import requests

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoConfig,              # <-- add
    GenerationConfig,        # <-- add
    pipeline,
)
import torch
import re

app = Flask(__name__)
CORS(app)   # this enables CORS for all routes

CLASSIFIER_DIR = "./classifier"
BART_DIR       = "./bart_paracomet"
LOCAL_SUMM_MODEL_DIR = "./bart_summarizer"

DEVICE = 0 if torch.cuda.is_available() else -1
DTYPE  = torch.float16 if torch.cuda.is_available() else None

EMBEDDER_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDER_NAME)
kw_model = KeyBERT(model=EMBEDDER_NAME)

# --- Classifier ---
clf_tok = AutoTokenizer.from_pretrained(CLASSIFIER_DIR, use_fast=True)
clf_mdl = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_DIR)
clf = pipeline("text-classification", model=clf_mdl, tokenizer=clf_tok, device=DEVICE,return_all_scores=True)

# --- BART generator ---
bart_tok = AutoTokenizer.from_pretrained(BART_DIR, use_fast=True)

bart_cfg = AutoConfig.from_pretrained(BART_DIR)
if getattr(bart_cfg, "early_stopping", None) is None:
    try:
        delattr(bart_cfg, "early_stopping")
    except Exception:
        bart_cfg.early_stopping = True
bart_cfg.use_cache = True
bart_cfg.attention_dropout = 0.0
bart_cfg.dropout = 0.0

bart_mdl = AutoModelForSeq2SeqLM.from_pretrained(
    BART_DIR,
    torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    config=bart_cfg,
)

# reload generation config with safer defaults
try:
    gen_cfg = GenerationConfig.from_pretrained(BART_DIR)
except Exception:
    gen_cfg = GenerationConfig()
gen_cfg.early_stopping = True
gen_cfg.num_beams = 1              # greedy
gen_cfg.do_sample = False
gen_cfg.max_new_tokens = 48        # was 64 → shorter, faster
gen_cfg.min_new_tokens = 8
gen_cfg.no_repeat_ngram_size = 0   # 0 is fastest
gen_cfg.length_penalty = 1.0

bart_mdl.generation_config = gen_cfg
try:
    bart_mdl = bart_mdl.to_bettertransformer()
except Exception:
    pass
bart = pipeline(
    "text2text-generation",
    model=bart_mdl,
    tokenizer=bart_tok,
    device=DEVICE,   # 0 if GPU
)



def download_bart_large_cnn(local_dir=LOCAL_SUMM_MODEL_DIR):
    # 1) Prefer safetensors
    snapshot_download(
        repo_id="facebook/bart-large-cnn",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "merges.txt",
            "vocab.json",
            "model.safetensors",
        ],
        ignore_patterns=["flax_model.msgpack", "tf_model.h5", "*.onnx", "*.tflite"],
        resume_download=True,
    )
    if not glob.glob(os.path.join(local_dir, "model.safetensors")):
        # 2) Fallback to PyTorch .bin if safetensors not available
        snapshot_download(
            repo_id="facebook/bart-large-cnn",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "merges.txt",
                "vocab.json",
                "pytorch_model.bin",
            ],
            ignore_patterns=["flax_model.msgpack", "tf_model.h5", "*.onnx", "*.tflite"],
            resume_download=True,
        )

download_bart_large_cnn()

# Load from the same folder
sum_tok = AutoTokenizer.from_pretrained(LOCAL_SUMM_MODEL_DIR, use_fast=False)
sum_mdl = AutoModelForSeq2SeqLM.from_pretrained(
    LOCAL_SUMM_MODEL_DIR,
    torch_dtype=torch.float32  # keep fp32 while stabilizing
)
# ---------- FAST summarization settings ----------
SUM_FAST = True                  # flip to False if you want max quality
SUM_BS   = 24                    # try 16–32 depending on VRAM
SUM_MAX  = 200 if not SUM_FAST else 160
SUM_MIN  =  50 if not SUM_FAST else  40
SUM_BEAMS= 4  if not SUM_FAST else 1   # beams=1 (greedy) is MUCH faster

# move summarizer to FP16 once stable
if torch.cuda.is_available():
    sum_mdl = sum_mdl.half().to("cuda:0").eval()   # <<< speedup
else:
    sum_mdl = sum_mdl.eval()


 
def is_url(text: str) -> bool:
    return re.match(r'https?://', text.strip()) is not None


# set up once (requires reddit app id/secret)
reddit = praw.Reddit(
    client_id="CyhbeigVD2af3tGzXpNmCw",
    client_secret="Q9tuDT2fyE9SIPoamYVwlPPH0RynDA",
    user_agent="my_bot by u/Acceptable_Panic_874"
)

# def get_reddit_comments(url, limit=10000):
#     submission = reddit.submission(url=url)
#     submission.comments.replace_more(limit=0)
#     comments = [c.body for c in submission.comments.list()[:limit]]
#     return comments      


def get_reddit_comments(url, limit=10000):
    if not url.endswith(".json"):
        if url.endswith("/"):
            url = url[:-1]
        url = url + ".json"

    resp = requests.get(url, headers={"User-agent": "Mozilla/5.0"})
    if resp.status_code != 200:
        return []

    data = resp.json()
    # comments are in data[1]["data"]["children"]
    comments = []
    for child in data[1]["data"]["children"]:
        body = child["data"].get("body")
        if body and body not in ("[deleted]", "[removed]"):
            comments.append(body)
        if len(comments) >= limit:
            break
    return comments
 
def predict_batch(texts, top_k=4, bs=128):
    """
    GPU batched classification using Dataset.map (returns top_k).
    """
    ds = Dataset.from_dict({"text": texts})
    def _map_cls(batch):
        with torch.inference_mode():
            outs = clf(batch["text"], truncation=True, batch_size=bs, return_all_scores=True)
        preds = []
        for scores in outs:
            top = sorted(scores, key=lambda x: x["score"], reverse=True)[:top_k]
            preds.append([
                {"emotion": s["label"].lower(), "confidence": round(float(s["score"]), 4)}
                for s in top
            ])
        return {"predictions": preds}
    ds = ds.map(_map_cls, batched=True, batch_size=bs)
    # Return same schema you used before
    return [{"text": t, "predictions": p} for t, p in zip(ds["text"], ds["predictions"])]



# def generate_one(text: str):
#     out = bart(
#         text,
#         truncation=True,
#         max_new_tokens=64,
#         min_new_tokens=8,
#         num_beams=4,
#         no_repeat_ngram_size=4,
#         length_penalty=1.0,
#         early_stopping=True,
#         # return_full_text=False,  # <- REMOVE THIS LINE
#     )[0]["generated_text"].strip()
#     return {"text": text, "generated": out}

def generate_batch(texts, bs=128):
    outs = bart(
        texts,
        truncation=True,
        max_length=512,   # shorten input
        max_new_tokens=48,
        min_new_tokens=8,
        num_beams=1,
        do_sample=False,
        early_stopping=True,
        batch_size=bs,
    )
    return [{"text": t, "generated": o["generated_text"].strip()} for t, o in zip(texts, outs)]


def move_to_cuda(model):
    if next(model.parameters()).is_cuda:
        return
    model.to("cuda:0")
    torch.cuda.empty_cache()

def move_to_cpu(model):
    if not next(model.parameters()).is_cuda:
        return
    model.to("cpu")
    torch.cuda.empty_cache()

def _tokenize_chunks(chunks, max_input_tokens=1024):
    enc = sum_tok(
        chunks,
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
        return_tensors="pt"
    )
    return {k: v.to("cuda:0") for k, v in enc.items()}

@torch.inference_mode()
def _summarize_chunks_fast(chunks, max_len, min_len, bs):
    if not chunks:
        return []
    # process in mini-batches to keep GPU full
    outs = []
    for i in range(0, len(chunks), bs):
        batch = chunks[i:i+bs]
        enc = _tokenize_chunks(batch)
        gen = sum_mdl.generate(
            **enc,
            max_new_tokens=max_len,          # use new-tokens budget
            min_new_tokens=min_len,
            do_sample=False,                  # deterministic
            num_beams=SUM_BEAMS,              # 1 in FAST mode
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
            no_repeat_ngram_size=3 if SUM_FAST else 4,
        )
        decoded = sum_tok.batch_decode(gen, skip_special_tokens=True)
        outs.extend([d.strip() for d in decoded])
    return outs

def chunk_text(text, max_words=2400):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

def safe_summarize_gpu(text: str, max_len=None, min_len=None, bs=None):
    max_len = SUM_MAX if max_len is None else max_len
    min_len = SUM_MIN if min_len is None else min_len
    bs      = SUM_BS  if bs is None      else bs

    chunks = list(chunk_text(text, max_words=750))
    summaries = _summarize_chunks_fast(chunks, max_len=max_len, min_len=min_len, bs=bs)
    return " ".join(summaries)


def summarize_topics_exact(texts):
    """
    GPU-only:
      - KeyBERT keywords
      - ST embeddings + KMeans
      - Summarize each cluster with Dataset.map batching
      - Fuse with another GPU summarization pass (Dataset.map)
    """

    comments = [t for t in texts if isinstance(t, str) and t.strip()]
    if not comments:
        return {"error": "No valid texts to summarize."}

    # Single comment fast-path
    if len(comments) == 1:
        final_summary = safe_summarize_gpu(comments[0], max_len=120, min_len=40, bs=128).replace(". ", ", ")
        return {"summary": final_summary, "mode": "single", "keywords": [], "cluster_summaries": []}

    # Keywords
    keywords = kw_model.extract_keywords(
        " ".join(comments),
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=5
    )
    topic_keywords = [kw for kw, _ in keywords] or ["topic"]

    # Embeddings (GPU) + KMeans
    emb = embedder.encode(comments, convert_to_tensor=True, batch_size=128, normalize_embeddings=False, show_progress_bar=False)
    emb_np = emb.detach().cpu().numpy()
    n_clusters = max(1, min(len(topic_keywords), len(comments)))
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(emb_np)

    # Summarize each cluster via Dataset batching
    cluster_summaries = []
    for cid in sorted(set(labels)):
        cluster_text = " ".join([comments[i] for i in range(len(comments)) if labels[i] == cid])
        topic_keyword = topic_keywords[cid] if cid < len(topic_keywords) else f"Topic {cid+1}"
        cluster_summary = safe_summarize_gpu(cluster_text, max_len=120, min_len=40, bs=128)
        cluster_summaries.append(f"{topic_keyword}: {cluster_summary}")

    fusion_prompt = (
        "Summarize the following customer feedback grouped by topics into a single coherent paragraph.\n"
        "Cover strengths and weaknesses without repetition. Focus on main patterns and insights.\n\n"
        + " ".join(cluster_summaries)
    )
    final_summary = safe_summarize_gpu(
        fusion_prompt,
        max_len=220 if not SUM_FAST else 160,
        min_len= 90 if not SUM_FAST else  60,
        bs=128
    ).replace(". ", ", ")

    return final_summary




# -------- unified pipeline: generate -> classify(generated) -> summarize(originals) --------
def run_full_pipeline(texts: list[str]):
    if len(texts) == 1 and is_url(texts[0]):
        url = texts[0]
        if "reddit.com" in url:
            texts = get_reddit_comments(url, limit=5000)
            if not texts:
                return {"success": False, "error": "No comments found at Reddit URL."}
    # 1) Generate implicit meanings (GPU batched)
    move_to_cuda(bart_mdl)
    move_to_cpu(sum_mdl)          # <-- frees ~1.6–2.8 GB
    gen_items = generate_batch(texts, bs=128)
    generated_texts = [it["generated"] for it in gen_items]

    # 2) Classify generated texts (GPU batched)
    cls_items = predict_batch(generated_texts, top_k=4, bs=128)

    move_to_cpu(bart_mdl)   
    move_to_cuda(sum_mdl)
              # <-- frees ~1.6–2.8 GB
    # 3) Summarize topics on original texts (GPU batched + chunking)
    summary = summarize_topics_exact(texts)

    # 4) Merge per-item outputs with best emotion
    merged = []
    best_emotions = []
    for original, gen_item, cls_item in zip(texts, gen_items, cls_items):
        preds = cls_item["predictions"] or []
        best = max(preds, key=lambda p: p["confidence"]) if preds else None
        if best:
            best_emotions.append(best["emotion"])
        merged.append({
            "original": original,
            "generated": gen_item["generated"],
            "classification": best,
        })

    # 5) Aggregate distribution of best emotions
    total = len(best_emotions) or 1
    counts = Counter(best_emotions)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:4]
    distribution = {emo: round((cnt / total) * 100, 2) for emo, cnt in sorted_counts}

    # 6) Representative items (≤4)
    items, chosen = [], set()
    for emo, _ in sorted_counts:
        for m in merged:
            if m["classification"] and m["classification"]["emotion"] == emo and emo not in chosen:
                items.append(m)
                chosen.add(emo)
                break

    return {
        "success": True,
        "count": len(texts),
        "items": items,
        "summary": summary,
        "distribution": distribution
    }



@app.route("/analyze_reviews", methods=["POST"])
def analyze_reviews_route():
    """
    Body: {"texts": ["...", "...", ...]}
    Flow:
      1) generate implicit meanings for each text
      2) classify the generated texts
      3) summarize topics on original texts
    """
    try:
        data = request.get_json(silent=True) or {}
        texts = data.get("texts")
        texts = [t.strip().strip('"') for t in texts if isinstance(t, str) and t.strip()]
        print("the texts we got ",texts)
        if not isinstance(texts, list) or not texts:
            return jsonify({"error": "Provide JSON with {'texts': [...]} (non-empty list)"}), 400

        # Optional trimming: drop empty/whitespace-only entries
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            return jsonify({"error": "No valid strings in 'texts'"}), 400

        result = run_full_pipeline(texts)
        print("the results", result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Body: {"texts": ["...", "..."]}
    """
    try:
        data = request.get_json(silent=True) or {}
        texts = data.get("texts")
        if not isinstance(texts, list):
            return jsonify({"error": "Provide JSON with {'texts': [...]}"}), 400
        results = predict_batch(texts)
        return jsonify({"success": True, "count": len(results), "predictions": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/generate", methods=["POST"])
def generate_route():
    """
    Body: {"texts": ["review [SEP] obs2", ...]}
    """
    try:
        data = request.get_json(silent=True) or {}
        texts = data.get("texts")
        if not isinstance(texts, list):
            return jsonify({"error": "Provide JSON with {'texts': [...]}"}), 400
        results = generate_batch(texts)
        return jsonify({"success": True, "count": len(results), "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/summarize_topics", methods=["POST"])
def summarize_topics_route():
    try:
        data = request.get_json(silent=True) or {}
        texts = data.get("texts")
        if not isinstance(texts, list):
            return jsonify({"error": "Provide JSON with {'texts': [...]}"}), 400
        result = summarize_topics_exact(texts)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/_debug_summarizer", methods=["POST"])
def _debug_summarizer():
    txt = request.get_json().get("text", "")
    return jsonify(summarize_topics_exact([txt]))

if __name__ == "__main__":
    # For production prefer: gunicorn -w 2 -k gthread --threads 8 -b 0.0.0.0:5000 app:app
    app.run(host="0.0.0.0", port=5000, debug=False)


#curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json"  -d "{\"texts\": [\"I am so happy today!\", \"This is terrible news\", \"What a surprise!\"]}"
#curl -X POST http://127.0.0.1:5000/generate -H "Content-Type: application/json" -d "{\"texts\": [\"The product arrived late [SEP]\", \"Great service [SEP]\"]}"
#curl -X POST http://127.0.0.1:5000/generate -H "Content-Type: application/json" -d "{\"texts\": [\"The product arrived late\", \"Great service\"]}"


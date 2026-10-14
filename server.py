import os
import sys
import json
import io
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from text_miner import MedicalTextMiner

# UTF-8 stdout setup for Windows
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "medical_transcriptions.csv")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder="static", static_url_path="")

# Safe integer helper
def safe_int(val, default=0):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default

# Initialize NLP & ML Engine
miner = MedicalTextMiner()
df_corpus = None
corpus_analytics_cache = None

if os.path.exists(CSV_PATH):
    print(f"[INFO] Initializing engine with dataset: {CSV_PATH} ...")
    df_corpus = pd.read_csv(CSV_PATH)
    df_corpus = df_corpus[df_corpus['transcription'].notna()].copy()
    df_corpus['medical_specialty'] = df_corpus['medical_specialty'].astype(str).str.strip()
    df_corpus['sample_name'] = df_corpus['sample_name'].astype(str).str.strip()
    miner.build_ir_index(df_corpus)
    miner.train_classifier(df_corpus)
    print(f"[SUCCESS] Loaded {len(df_corpus)} records, IR Index and Naive Bayes Classifier ready with {len(miner.classifier_classes)} specialties!")
else:
    print(f"[WARNING] Dataset not found at {CSV_PATH}")

@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/meta", methods=["GET"])
def get_meta():
    """Returns dataset overview, total records, specialties, and top category distribution."""
    if df_corpus is None:
        return jsonify({"status": "error", "message": "Corpus not loaded"}), 500

    total_records = int(len(df_corpus))
    specialties = sorted(df_corpus['medical_specialty'].unique().tolist())
    counts = df_corpus['medical_specialty'].value_counts()
    
    top_specialties = [
        {"name": k, "count": int(v)} for k, v in counts.head(35).items()
    ]

    return jsonify({
        "status": "success",
        "total_records": total_records,
        "specialties_count": len(specialties),
        "specialties": specialties,
        "top_specialties": top_specialties
    })

@app.route("/api/records", methods=["GET"])
def get_records():
    """Returns samples filtered by specialty or search query with safe limit handling."""
    if df_corpus is None:
        return jsonify({"status": "error", "message": "Corpus not loaded"}), 500

    specialty = request.args.get("specialty", "").strip()
    query = request.args.get("query", "").strip()
    limit = max(1, min(100, safe_int(request.args.get("limit", 20), 20)))

    filtered = df_corpus
    if specialty:
        filtered = filtered[filtered['medical_specialty'] == specialty]
    if query:
        filtered = filtered[filtered['transcription'].str.contains(query, case=False, na=False)]

    results = []
    seen = set()
    for _, row in filtered.iterrows():
        s_name = str(row['sample_name']).strip()
        t_sig = " ".join(str(row['transcription'])[:100].lower().split())
        if s_name in seen or t_sig in seen:
            continue
        seen.add(s_name)
        seen.add(t_sig)
        results.append({
            "specialty": row['medical_specialty'],
            "sample_name": s_name,
            "snippet": str(row['transcription'])[:200] + "...",
            "text": str(row['transcription'])
        })
        if len(results) >= limit:
            break

    return jsonify({
        "status": "success",
        "count": len(results),
        "records": results
    })

@app.route("/api/analyze", methods=["POST"])
def analyze_text():
    """Deep text mining: Multi-Algorithm Summarization, NER, Acuity Triage, ROUGE evaluation."""
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    num_sentences = max(1, min(15, safe_int(data.get("num_sentences", 4), 4)))
    method = data.get("method", "TF-IDF Extractive")

    if not text:
        return jsonify({"status": "error", "message": "Text is required"}), 400

    summary, sentences, top_keywords, rouge, entities, lexical_stats, acuity = miner.summarize(
        text, num_sentences=num_sentences, method=method
    )

    pred_spec, spec_probs = miner.predict_specialty(text)

    return jsonify({
        "status": "success",
        "summary": summary,
        "sentence_count": len(sentences),
        "top_keywords": top_keywords,
        "rouge": rouge,
        "entities": entities,
        "lexical_stats": lexical_stats,
        "acuity": acuity,
        "predicted_specialty": pred_spec,
        "specialty_probabilities": spec_probs
    })

@app.route("/api/search", methods=["POST"])
def search_ir():
    """Information Retrieval Cosine Similarity search over corpus with optional specialty filter."""
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    specialty = data.get("specialty", "").strip()
    top_k = max(1, min(30, safe_int(data.get("top_k", 6), 6)))

    if not query:
        return jsonify({"status": "error", "message": "Query is required"}), 400

    results = miner.search_similar_records(query, specialty=specialty if specialty else None, top_k=top_k)
    return jsonify({
        "status": "success",
        "results": results
    })

@app.route("/api/classify", methods=["POST"])
def classify_specialty():
    """Supervised Specialty Classifier over ~33 departments."""
    data = request.get_json() or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"status": "error", "message": "Text is required"}), 400

    pred, probs = miner.predict_specialty(text)
    return jsonify({
        "status": "success",
        "prediction": pred,
        "probabilities": probs
    })

@app.route("/api/upload", methods=["POST"])
def upload_document():
    """Receives PDF or TXT clinical file and returns extracted narrative text."""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files['file']
    filename = file.filename or ""

    try:
        if filename.lower().endswith('.pdf'):
            text = MedicalTextMiner.extract_text_from_pdf(io.BytesIO(file.read()))
        else:
            # Plain text file
            text = file.read().decode('utf-8', errors='replace')

        text = text.strip()
        if not text:
            return jsonify({"status": "error", "message": "File was empty or no text could be extracted"}), 400

        words = len(text.split())
        return jsonify({
            "status": "success",
            "filename": filename,
            "text": text,
            "word_count": words
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Document parsing error: {str(e)}"}), 500

@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    """Computes corpus-wide clinical intelligence analytics: word counts, acuity risk distribution."""
    global corpus_analytics_cache
    if corpus_analytics_cache is not None:
        return jsonify(corpus_analytics_cache)

    if df_corpus is None:
        return jsonify({"status": "error", "message": "Corpus not loaded"}), 500

    # Compute word counts across corpus
    word_counts = df_corpus['transcription'].str.split().str.len()
    avg_words = int(word_counts.mean())
    max_words = int(word_counts.max())
    min_words = int(word_counts.min())

    # Sample 300 notes for corpus acuity distribution
    sample_df = df_corpus.sample(n=min(300, len(df_corpus)), random_state=42)
    acuity_counts = {"Routine / Low Acuity": 0, "Moderate Acuity": 0, "High / Critical Risk": 0}
    for _, row in sample_df.iterrows():
        acu = miner.evaluate_clinical_acuity(str(row['transcription']))
        lvl = acu.get('level', 'Routine / Low Acuity')
        acuity_counts[lvl] = acuity_counts.get(lvl, 0) + 1

    corpus_analytics_cache = {
        "status": "success",
        "total_records": len(df_corpus),
        "total_words": int(word_counts.sum()),
        "avg_words_per_record": avg_words,
        "max_words": max_words,
        "min_words": min_words,
        "acuity_distribution": acuity_counts
    }
    return jsonify(corpus_analytics_cache)

@app.route("/api/batch", methods=["POST"])
def batch_summarize():
    """Batch summarization across medical departments with safe count."""
    if df_corpus is None:
        return jsonify({"status": "error", "message": "Corpus not loaded"}), 500

    data = request.get_json() or {}
    specialty = data.get("specialty", "").strip()
    count = max(1, min(15, safe_int(data.get("count", 3), 3)))

    subset = df_corpus
    if specialty:
        subset = df_corpus[df_corpus['medical_specialty'] == specialty]

    results = []
    for _, row in subset.head(count).iterrows():
        raw_text = str(row['transcription']).strip()
        summ, _, _, rouge, entities, stats, acuity = miner.summarize(raw_text, num_sentences=3)
        results.append({
            "sample_name": row['sample_name'],
            "specialty": row['medical_specialty'],
            "original_words": stats['original_word_count'],
            "summary_words": stats['summary_word_count'],
            "compression": stats['compression_ratio'],
            "rouge1": rouge.get("ROUGE-1", 0),
            "acuity_level": acuity['level'],
            "acuity_color": acuity['color'],
            "summary": summ,
            "vitals_bp": entities.get("vitals_bp", []),
            "dosages": entities.get("dosages", [])
        })

    return jsonify({
        "status": "success",
        "processed": len(results),
        "results": results
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=================================================================")
    print(f"       🏥 MediMine AI — Native 3D Clinical Platform Server       ")
    print(f"       Running on: http://0.0.0.0:{port}                        ")
    print(f"=================================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)

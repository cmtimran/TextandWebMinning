import os
import sys
import pandas as pd
from text_miner import MedicalTextMiner

# UTF-8 stdout setup for Windows terminal
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 65)
print("=== Medical Text Mining, NER & TF-IDF Summarization Engine ===")
print("=" * 65)

# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_path = os.path.join(DATA_DIR, "medical_transcriptions.csv")
output_file_path = os.path.join(OUTPUT_DIR, "summary_output.txt")

# ১. ডেটাসেট লোড
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found at {csv_path}!")

print(f"\n[INFO] Loading dataset from: {csv_path}")
df = pd.read_csv(csv_path)
valid_data = df[df["transcription"].notna()].iloc[0]
specialty = str(valid_data.get("medical_specialty", "General")).strip()
sample_name = str(valid_data.get("sample_name", "Clinical Record")).strip()
medical_text = str(valid_data["transcription"]).strip()

print(f"-> Selected Category : {specialty}")
print(f"-> Record Title     : {sample_name}")

print("\n--- Original Clinical Record (Snippet) ---")
print(medical_text[:350] + ("..." if len(medical_text) > 350 else ""))

# ২. টেক্সট মাইনিং পাইপলাইন এক্সিকিউশন
miner = MedicalTextMiner()
summary, sentences, top_keywords, rouge_metrics, entities, lexical_stats, acuity = miner.summarize(
    medical_text, num_sentences=4
)

# ৩. ফলাফল প্রদর্শন
print(f"\n[INFO] Sentences Analyzed: {len(sentences)}")
print(f"[INFO] Compression Ratio: {lexical_stats.get('compression_ratio')}")
print(f"[INFO] Clinical Acuity  : {acuity.get('level')} (Score: {acuity.get('score')}/100)")

print("\n--- Extracted Medical Entities (NER) ---")
for entity_type, matches in entities.items():
    if matches:
        print(f"  * {entity_type.replace('_', ' ').title()}: {', '.join(set(matches))}")

print("\n--- Top Extracted Medical Keywords (TF-IDF) ---")
for kw, score in list(top_keywords.items())[:6]:
    print(f"  * {kw:<15} (Score: {score:.3f})")

print("\n--- Extractive Summary (TF-IDF Ranked) ---")
print(summary)

print("\n--- Evaluation Metrics (ROUGE Scores) ---")
for metric, score in rouge_metrics.items():
    print(f"  * {metric:<8}: {score}%")

# ৪. আউটপুট ফাইলে বিস্তারিত সেভ করা
with open(output_file_path, "w", encoding="utf-8") as out:
    out.write(f"Category : {specialty}\n")
    out.write(f"Title    : {sample_name}\n\n")
    out.write("--- Extracted Clinical Entities ---\n")
    for k, v in entities.items():
        if v:
            out.write(f"{k.title()}: {', '.join(set(v))}\n")
    out.write("\n--- Top TF-IDF Keywords ---\n")
    out.write(", ".join(top_keywords.keys()) + "\n\n")
    out.write("--- Clinical Summary ---\n")
    out.write(summary + "\n\n")
    out.write("--- ROUGE Performance ---\n")
    for k, v in rouge_metrics.items():
        out.write(f"{k}: {v}%\n")

print(f"\n[Success] Full report generated and saved to '{output_file_path}'!\n")
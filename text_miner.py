import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from rouge_score import rouge_scorer
import pypdf

class MedicalTextMiner:
    """
    Expert Medical Text Mining & Clinical Intelligence Engine:
    1. Robust Clinical Sentence Tokenization (Abbreviation-Aware)
    2. Deep Clinical Entity Recognition (NER: BP, HR, Temp, SpO2, RR, Allergies, Weight, Dosages, Demographics)
    3. Multi-Algorithm Summarization:
       - TF-IDF Salience (Length-Normalized)
       - TextRank (Graph PageRank Centrality)
       - Lead-3 / Lead-N Clinical Baseline
    4. Clinical Acuity & Urgency Risk Scoring (Sepsis, Stroke, CVA, Anaphylaxis, Hemorrhage)
    5. Information Retrieval (IR) Vector Space Model with Cosine Similarity & Query Term Highlighting
    6. Supervised Medical Specialty Classification (Full-Spectrum ~33 Specialties)
    7. PDF & Clinical Document Parsing
    8. Quantitative Evaluation Metrics (ROUGE-1, ROUGE-2, ROUGE-L, Lexical Compression)
    """

    # Common clinical and general abbreviations that should not break sentences
    ABBREVIATIONS = (
        r'(?:Dr|Mr|Mrs|Ms|vs|approx|e\.g|i\.e|no|pt|yo|y/o|q\.d|b\.i\.d|t\.i\.d|q\.i\.d|'
        r'p\.r\.n|Tab|Cap|IV|IM|po|mg|mcg|ml|kg|lbs|oz|vol|dept|div|hr|min|sec|wt|ht)'
    )

    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.ir_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            sublinear_tf=True
        )
        self.classifier = None
        self.classifier_classes = []
        self.ir_matrix = None
        self.corpus_df = None

    @classmethod
    def split_sentences(cls, text: str):
        """
        Abbreviation-aware clinical sentence tokenization.
        Protects abbreviations like 'Dr.', 'vs.', 'Tab.', 'mg.' from being split.
        """
        if not text:
            return []

        cleaned = re.sub(r'\s+', ' ', text).strip()
        # Protect abbreviations by temporarily replacing period with placeholder
        def protect_abbr(match):
            return match.group(0).replace('.', '__DOT__')

        protected = re.sub(r'\b' + cls.ABBREVIATIONS + r'\.', protect_abbr, cleaned, flags=re.IGNORECASE)
        # Also protect numbers like 120.5 or 3.5
        protected = re.sub(r'(\d+)\.(\d+)', r'\1__DOT__\2', protected)

        # Split on sentence boundaries
        raw_sentences = re.split(r'(?<=[.!?])\s+', protected)

        sentences = []
        for s in raw_sentences:
            restored = s.replace('__DOT__', '.').strip()
            # Retain clinical sentences if they contain at least 2 words and 5 characters
            words = restored.split()
            if len(words) >= 2 and len(restored) >= 5:
                sentences.append(restored)

        return sentences if sentences else ([cleaned] if len(cleaned) > 5 else [])

    @staticmethod
    def extract_text_from_pdf(file_bytes):
        """Extracts text content from uploaded PDF file."""
        reader = pypdf.PdfReader(file_bytes)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()

    def build_ir_index(self, df):
        """Builds TF-IDF vector matrix across all documents for Cosine Similarity search."""
        self.corpus_df = df.copy()
        transcriptions = self.corpus_df['transcription'].fillna("").astype(str).tolist()
        self.ir_matrix = self.ir_vectorizer.fit_transform(transcriptions)

    def search_similar_records(self, query: str, specialty: str = None, top_k: int = 6):
        """
        Information Retrieval engine using TF-IDF Vector Space Model & Cosine Similarity.
        Supports optional specialty filtering and query term extraction for highlighting.
        """
        if self.ir_matrix is None or self.corpus_df is None or not query.strip():
            return []

        query_vec = self.ir_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.ir_matrix).flatten()

        # Extract search terms for frontend highlighting
        search_terms = list(set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())))

        candidate_indices = similarities.argsort()[::-1][:max(top_k * 6, 60)]

        results = []
        seen_names = set()
        seen_texts = set()

        for idx in candidate_indices:
            if len(results) >= top_k:
                break
            score = float(similarities[idx])
            if score <= 0.01:
                break
            row = self.corpus_df.iloc[idx]

            # Specialty filtering if provided
            if specialty and str(row.get('medical_specialty', '')).strip().lower() != specialty.strip().lower():
                continue

            raw_text = str(row['transcription']).strip()
            sample_name = str(row['sample_name']).strip()

            text_signature = " ".join(raw_text[:100].lower().split())
            if sample_name.lower() in seen_names or text_signature in seen_texts:
                continue

            seen_names.add(sample_name.lower())
            seen_texts.add(text_signature)

            entities = self.extract_medical_entities(raw_text)
            acuity = self.evaluate_clinical_acuity(raw_text)

            results.append({
                "score": round(score * 100, 1),
                "specialty": row['medical_specialty'],
                "sample_name": sample_name,
                "snippet": (raw_text[:280] + "...") if len(raw_text) > 280 else raw_text,
                "full_text": raw_text,
                "word_count": len(raw_text.split()),
                "entities": entities,
                "acuity": acuity,
                "matched_terms": search_terms
            })
        return results

    def train_classifier(self, df):
        """
        Trains a high-performance Multinomial Naive Bayes classifier.
        Expands coverage to all medical specialties with >= 10 transcription records (~33 departments).
        """
        counts = df['medical_specialty'].value_counts()
        eligible_specialties = counts[counts >= 10].index.tolist()
        filtered = df[df['medical_specialty'].isin(eligible_specialties)].dropna(subset=['transcription'])

        self.classifier = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english',
                max_features=4000,
                sublinear_tf=True
            )),
            ('clf', MultinomialNB(alpha=0.15))
        ])
        self.classifier.fit(filtered['transcription'], filtered['medical_specialty'])
        self.classifier_classes = list(self.classifier.classes_)

    def predict_specialty(self, text: str):
        """Predicts the medical specialty/department with sorted probability distribution."""
        if self.classifier is None or not text.strip():
            return "General Medicine", {}

        try:
            pred = self.classifier.predict([text])[0]
            probs = self.classifier.predict_proba([text])[0]
            classes = self.classifier.classes_

            prob_dict = {classes[i]: round(probs[i] * 100, 1) for i in range(len(classes))}
            sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
            return pred, sorted_probs
        except Exception:
            return "General Medicine", {}

    def evaluate_clinical_acuity(self, text: str):
        """
        Calculates patient clinical risk & urgency level based on emergency clinical indicators.
        Detects acute critical conditions (sepsis, stroke, respiratory distress, shock).
        """
        critical_keywords = [
            "acute", "emergency", "arrest", "severe", "hemorrhage", "infarction",
            "respiratory distress", "unstable", "critical", "hypotension", "tachycardia",
            "ischemia", "failure", "coma", "shock", "malignant", "sepsis", "septic shock",
            "stroke", "cva", "anaphylaxis", "cyanosis", "hypoxia", "intubation", "ventilator",
            "syncope", "dka", "arrhythmia", "ventricular fibrillation", "hemiplegia"
        ]
        moderate_keywords = [
            "chronic", "pain", "elevated", "edema", "hypertension", "infection",
            "abnormal", "recurrent", "fatigue", "fever", "nausea", "dyspnea",
            "palpitations", "wheezing", "angina", "lesion", "mass"
        ]

        text_lower = text.lower()
        crit_matches = []
        for kw in critical_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                crit_matches.append(kw)

        mod_matches = []
        for kw in moderate_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                mod_matches.append(kw)

        score = min(100, (len(crit_matches) * 22) + (len(mod_matches) * 8))
        if score >= 50 or any(c in crit_matches for c in ["arrest", "shock", "infarction", "sepsis", "stroke", "coma"]):
            level = "High / Critical Risk"
            color = "#EF4444"
        elif score >= 20:
            level = "Moderate Acuity"
            color = "#F59E0B"
        else:
            level = "Routine / Low Acuity"
            color = "#10B981"

        return {
            "score": score,
            "level": level,
            "color": color,
            "critical_matches": crit_matches,
            "moderate_matches": mod_matches
        }

    def extract_medical_entities(self, text: str):
        """
        Deep clinical entity extraction (NER):
        - Blood Pressure (BP)
        - Heart Rate / Pulse (HR)
        - Body Temperature (Temp)
        - Oxygen Saturation (SpO2)
        - Respiration Rate (RR)
        - Body Weight & Mass
        - Prescriptions, Medications & Dosages
        - Patient Demographics
        - Allergies & Drug Reactions (NKDA)
        """
        # Blood pressure (e.g. 120/80, 140 / 90 mmHg)
        bp_matches = re.findall(r'\b\d{2,3}\s*/\s*\d{2,3}(?:\s*mmHg)?\b', text, re.IGNORECASE)
        vitals_bp = [re.sub(r'\s+', '', b) for b in bp_matches]

        # Heart rate / pulse
        hr_matches = re.findall(
            r'\b(?:HR|pulse|heart[\s-]*rate)[:\s]*(?:is|was|of)?\s*(\d{2,3})\s*(?:bpm|beats\s*/\s*min|beats\s*per\s*minute)?\b',
            text, re.IGNORECASE
        )
        vitals_hr = [f"{h} bpm" for h in hr_matches]

        # Temperature (e.g. Temp 98.6 F, fever 101.4 F, 37.5 C, afebrile)
        temp_matches = re.findall(
            r'\b(?:temp(?:erature)?|T|fever|febrile)[:\s]*(?:is|was|of|at)?\s*(\d{2,3}(?:\.\d+)?)\s*(?:°?\s*[FC]|degrees|Fahrenheit|Celsius)?\b',
            text, re.IGNORECASE
        )
        if not temp_matches and re.search(r'\bafebrile\b', text, re.IGNORECASE):
            temp_matches = ["Afebrile"]
        vitals_temp = [f"{t}°F" if re.match(r'^\d+(\.\d+)?$', str(t)) and float(t) > 70 else (f"{t}°C" if re.match(r'^\d+(\.\d+)?$', str(t)) and float(t) <= 45 else str(t)) for t in temp_matches]

        # Oxygen Saturation (SpO2)
        spo2_matches = re.findall(
            r'\b(?:SpO2|O2\s*sat(?:uration)?|pulse\s*ox(?:imetry)?)[:\s]*(?:is|was|of)?\s*(\d{2,3}%?(?:\s*on\s*[\w\s]+)?)\b',
            text, re.IGNORECASE
        )
        vitals_spo2 = [s.strip() for s in spo2_matches]

        # Respiration Rate (RR)
        rr_matches = re.findall(
            r'\b(?:RR|respirations?|respiratory\s*rate)[:\s]*(?:is|was|of)?\s*(\d{1,2})\s*(?:bpm|breaths?\s*/\s*min(?:ute)?)?\b',
            text, re.IGNORECASE
        )
        vitals_rr = [f"{r} breaths/min" for r in rr_matches]

        # Allergies
        allergy_matches = []
        if re.search(r'\b(?:NKDA|No\s*known\s*(?:drug\s*)?allergies)\b', text, re.IGNORECASE):
            allergy_matches.append("NKDA (No Known Drug Allergies)")
        allergy_found = re.findall(r'\b(?:allergies|allergic\s*to)[:\s]*([^\n.;]{3,60})', text, re.IGNORECASE)
        for a in allergy_found:
            clean_a = a.strip()
            if clean_a.lower() not in ["none", "no", "denies", "nkda", "no known drug allergies"]:
                allergy_matches.append(clean_a)

        # Body weight / mass
        vitals_weight = list(set(re.findall(r'\b\d{2,3}(?:\.\d+)?\s*(?:lbs?|pounds?|kg|kilograms?)\b', text, re.IGNORECASE)))

        # Dosages & Prescriptions
        dosages = list(set(re.findall(r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml|g|tablets?|sprays?|capsules?|pills?|units?)\b', text, re.IGNORECASE)))

        # Demographics
        demographics = list(set(re.findall(r'\b\d{1,2}[-\s]year[-\s]old\s+(?:male|female|man|woman|boy|girl)\b', text, re.IGNORECASE)))

        return {
            "vitals_bp": list(set(vitals_bp)),
            "vitals_hr": list(set(vitals_hr)),
            "vitals_temp": list(set(vitals_temp)),
            "vitals_spo2": list(set(vitals_spo2)),
            "vitals_rr": list(set(vitals_rr)),
            "vitals_weight": vitals_weight,
            "dosages": dosages[:10],
            "demographics": demographics,
            "allergies": list(set(allergy_matches))
        }

    def compute_lexical_stats(self, text: str, summary: str):
        """Quantitative lexical statistics & read-time metrics."""
        orig_words = re.findall(r'\b\w+\b', text)
        summ_words = re.findall(r'\b\w+\b', summary)
        orig_count = len(orig_words)
        summ_count = len(summ_words)
        compression_ratio = round((1 - (summ_count / orig_count)) * 100, 1) if orig_count > 0 else 0
        reading_time_orig = max(1, round(orig_count / 200, 1))
        reading_time_summ = max(1, round(summ_count / 200, 1))

        return {
            "original_word_count": orig_count,
            "summary_word_count": summ_count,
            "compression_ratio": f"{compression_ratio}%",
            "reading_time_saved_min": round(max(0, reading_time_orig - reading_time_summ), 1)
        }

    def _textrank_summarize(self, sentences: list, num_sentences: int):
        """
        Graph-based TextRank Extractive Summarization (PageRank Centrality).
        Builds sentence-to-sentence similarity graph and computes eigenvector centrality.
        """
        n = len(sentences)
        if n <= num_sentences:
            return " ".join(sentences)

        # Thread-safe local TF-IDF vectorizer
        vec = TfidfVectorizer(stop_words='english', max_features=1500)
        try:
            tfidf = vec.fit_transform(sentences)
            sim_matrix = cosine_similarity(tfidf, tfidf)

            # Eliminate self-loops and weak edges
            np.fill_diagonal(sim_matrix, 0)
            sim_matrix[sim_matrix < 0.05] = 0

            # Degree normalization
            row_sums = sim_matrix.sum(axis=1)
            row_sums[row_sums == 0] = 1.0
            norm_matrix = sim_matrix / row_sums[:, np.newaxis]

            # Power Iteration PageRank
            d = 0.85
            scores = np.ones(n) / n
            for _ in range(35):
                scores = (1 - d) / n + d * norm_matrix.T.dot(scores)

            top_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)[:num_sentences]
            top_indices.sort()
            return " ".join([sentences[i] for i in top_indices])
        except Exception:
            return " ".join(sentences[:num_sentences])

    def summarize(self, text: str, num_sentences: int = 4, method: str = "TF-IDF Extractive"):
        """
        Performs multi-algorithm clinical text mining & summarization:
        - TF-IDF Extractive (Length-Normalized)
        - TextRank Graph (PageRank Centrality)
        - Lead-3 / Lead-N Baseline
        """
        sentences = self.split_sentences(text)
        if not sentences:
            empty_entities = self.extract_medical_entities(text)
            empty_stats = self.compute_lexical_stats(text, "")
            empty_acuity = self.evaluate_clinical_acuity(text)
            return "", [], {}, {"ROUGE-1": 0.0, "ROUGE-2": 0.0, "ROUGE-L": 0.0}, empty_entities, empty_stats, empty_acuity

        top_keywords = {}
        # Thread-safe local vectorizer for per-call execution
        local_vec = TfidfVectorizer(stop_words='english', max_features=2500)
        tfidf_matrix = None
        try:
            tfidf_matrix = local_vec.fit_transform(sentences)
            feature_names = local_vec.get_feature_names_out()
            word_scores = tfidf_matrix.sum(axis=0).A1
            sorted_indices = word_scores.argsort()[::-1]
            top_keywords = {feature_names[i]: float(word_scores[i]) for i in sorted_indices[:15]}
        except Exception:
            words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', text)]
            from collections import Counter
            counts = Counter(words)
            top_keywords = {k: float(v) for k, v in counts.most_common(15)}

        if len(sentences) <= num_sentences:
            summary = " ".join(sentences)
        elif method == "Lead-3 / Lead-N Baseline":
            top_indices = list(range(min(num_sentences, len(sentences))))
            summary = " ".join([sentences[i] for i in top_indices])
        elif method == "TextRank Graph (PageRank)" or method == "TextRank":
            summary = self._textrank_summarize(sentences, num_sentences)
        else:
            # Default: TF-IDF Salience with Length Normalization
            if tfidf_matrix is not None:
                sentence_scores = tfidf_matrix.sum(axis=1).A1
                # Length normalization to prevent bias towards long run-on sentences
                for idx, s in enumerate(sentences):
                    word_len = max(1, len(s.split()))
                    sentence_scores[idx] = sentence_scores[idx] / (word_len ** 0.55)

                top_indices = sorted(
                    range(len(sentences)),
                    key=lambda idx: sentence_scores[idx],
                    reverse=True
                )[:num_sentences]
                top_indices.sort()
                summary = " ".join([sentences[i] for i in top_indices])
            else:
                summary = " ".join(sentences[:num_sentences])

        rouge_scores = self.scorer.score(text, summary)
        formatted_rouge = {
            "ROUGE-1": round(rouge_scores['rouge1'].fmeasure * 100, 2),
            "ROUGE-2": round(rouge_scores['rouge2'].fmeasure * 100, 2),
            "ROUGE-L": round(rouge_scores['rougeL'].fmeasure * 100, 2),
        }

        entities = self.extract_medical_entities(text)
        lexical_stats = self.compute_lexical_stats(text, summary)
        acuity = self.evaluate_clinical_acuity(text)

        return summary, sentences, top_keywords, formatted_rouge, entities, lexical_stats, acuity

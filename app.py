import os
from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from summarizer import Summarizer  # Fixed BertSummarizer import
import PyPDF2
import nltk
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu

# Ensure necessary nltk components are available
nltk.download("punkt")

app = Flask(__name__)

# Load models once for better performance
try:
    abstractive_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    extractive_summarizer = Summarizer()
except Exception as e:
    print(f"Model loading failed: {e}")
    abstractive_summarizer = None
    extractive_summarizer = None

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    except Exception as e:
        print(f"PDF extraction failed: {e}")
        return None  # Handle errors gracefully
    return text.strip()

def compute_rouge(reference, generated):
    """Computes ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {key: round(scores[key].fmeasure, 4) for key in scores}

def compute_bleu(reference, generated):
    """Computes BLEU score."""
    reference_tokens = [sent_tokenize(reference)]
    generated_tokens = sent_tokenize(generated)
    return round(sentence_bleu(reference_tokens, generated_tokens) * 100, 2)

def compute_precision_recall(reference, generated):
    """Computes Precision and Recall."""
    reference_words = set(word_tokenize(reference.lower()))
    generated_words = set(word_tokenize(generated.lower()))

    precision = len(generated_words.intersection(reference_words)) / len(generated_words) if generated_words else 0.0
    recall = len(generated_words.intersection(reference_words)) / len(reference_words) if reference_words else 0.0

    return {"Precision": round(precision, 4), "Recall": round(recall, 4)}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    text = ""
    summary_type = request.form.get("type") or request.json.get("type", "").lower()

    # Handle file or text input
    if "file" in request.files:
        pdf_file = request.files["file"]
        text = extract_text_from_pdf(pdf_file)
        if not text:
            return jsonify({"error": "Failed to extract text from the PDF."}), 400
    else:
        data = request.get_json() or {}
        text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Generate reference summary (first 3 sentences for evaluation)
    reference_summary = " ".join(sent_tokenize(text)[:3])

    try:
        summary_text = ""
        if summary_type == "extractive":
            if not extractive_summarizer:
                return jsonify({"error": "Extractive summarizer model not available."}), 500
            summary = extractive_summarizer(text, num_sentences=4)
            summary_text = " ".join(summary) if isinstance(summary, list) else summary
        elif summary_type == "abstractive":
            if not abstractive_summarizer:
                return jsonify({"error": "Abstractive summarizer model not available."}), 500
            if len(text.split()) < 50:
                return jsonify({"error": "Text too short for abstractive summarization. Provide at least 50 words."}), 400
            summary = abstractive_summarizer(text, max_length=180, min_length=80, do_sample=False)
            summary_text = summary[0]["summary_text"] if summary else "Summarization failed."
        else:
            return jsonify({"error": "Invalid summary type. Use 'extractive' or 'abstractive'."}), 400
    except Exception as e:
        return jsonify({"error": f"Summarization failed: {str(e)}"}), 500

    # Compute accuracy scores
    rouge_scores = compute_rouge(reference_summary, summary_text)
    bleu_score = compute_bleu(reference_summary, summary_text)
    precision_recall = compute_precision_recall(reference_summary, summary_text)

    # Calculate average score
    average_score = (
        rouge_scores["rouge1"] +
        rouge_scores["rouge2"] +
        rouge_scores["rougeL"] +
        (bleu_score / 100)
    ) / 4

    return jsonify({
        "summary": summary_text,
        "rouge_scores": rouge_scores,
        "bleu_score": bleu_score,
        "precision_recall": precision_recall,
        "score": round(average_score, 4)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Ensure correct port binding for Render
    app.run(host="0.0.0.0", port=port)

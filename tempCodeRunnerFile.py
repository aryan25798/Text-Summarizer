from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from summarizer import Summarizer
import PyPDF2
import nltk
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu

nltk.download('punkt')

app = Flask(__name__)

# Load summarization models
abstractive_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
extractive_summarizer = Summarizer()

def extract_text_from_pdf(pdf_file):
    """Extract text from an uploaded PDF file."""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    except Exception as e:
        return None  # Handle errors gracefully
    return text.strip()

def compute_rouge(reference, generated):
    """Compute ROUGE scores for text summarization."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {
        "ROUGE-1": scores["rouge1"].fmeasure,
        "ROUGE-2": scores["rouge2"].fmeasure,
        "ROUGE-L": scores["rougeL"].fmeasure
    }

def compute_bleu(reference, generated):
    """Compute BLEU score with sentence-level tokenization."""
    reference_tokens = [sent_tokenize(reference)]
    generated_tokens = sent_tokenize(generated)
    return sentence_bleu(reference_tokens, generated_tokens) * 100  # Convert to percentage

def compute_precision_recall(reference, generated):
    """Calculate Precision and Recall scores."""
    reference_words = set(word_tokenize(reference.lower()))
    generated_words = set(word_tokenize(generated.lower()))

    if len(generated_words) == 0:
        precision = 0.0
    else:
        precision = len(generated_words.intersection(reference_words)) / len(generated_words)

    if len(reference_words) == 0:
        recall = 0.0
    else:
        recall = len(generated_words.intersection(reference_words)) / len(reference_words)

    return {"Precision": precision, "Recall": recall}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = ""
    summary_type = request.form.get("type") or request.json.get("type", "").lower()

    # Handle file upload
    if 'file' in request.files:
        pdf_file = request.files['file']
        text = extract_text_from_pdf(pdf_file)
        if not text:
            return jsonify({"error": "Failed to extract text from the PDF."}), 400
    else:
        data = request.get_json() or {}
        text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Reference summary (for evaluation)
    reference_summary = " ".join(sent_tokenize(text)[:3])  # Use first 3 sentences as reference

    if summary_type == "extractive":
        summary = extractive_summarizer(text, num_sentences=4)  # Capture more details
        summary_text = " ".join(summary) if isinstance(summary, list) else summary
    elif summary_type == "abstractive":
        if len(text.split()) < 50:
            return jsonify({"error": "Text too short for abstractive summarization. Provide at least 50 words."}), 400
        summary = abstractive_summarizer(text, max_length=180, min_length=80, do_sample=False)
        summary_text = summary[0]['summary_text'] if summary else "Summarization failed."
    else:
        return jsonify({"error": "Invalid summary type. Use 'extractive' or 'abstractive'."}), 400

    # Compute accuracy scores
    rouge_scores = compute_rouge(reference_summary, summary_text)
    bleu_score = compute_bleu(reference_summary, summary_text)
    precision_recall = compute_precision_recall(reference_summary, summary_text)

    return jsonify({
        "summary": summary_text,
        "rouge_scores": rouge_scores,
        "bleu_score": bleu_score,
        "precision_recall": precision_recall
    })

if __name__ == '__main__':
    app.run(debug=True)

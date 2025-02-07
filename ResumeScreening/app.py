import os
import pickle
import docx
import pdfplumber
from flask import Flask, render_template, request

# ML Model & Vectorizer Load Karne Ka Function
def load_model_and_vectorizer():
    model_path = "resume_screening.pkl"
    vectorizer_path = "tfidf_vectorizer.pkl"

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        with open(model_path, "rb") as model_file:
            rf = pickle.load(model_file)
        with open(vectorizer_path, "rb") as vectorizer_file:
            tfidf = pickle.load(vectorizer_file)

        print("[INFO] Model and Vectorizer loaded successfully.")
        print("Model:", rf)
        print("Vectorizer:", tfidf)

        return rf, tfidf
    else:
        print("[ERROR] Model or Vectorizer missing!")
        return None, None

    
def process_resume(file):
    rf, tfidf = load_model_and_vectorizer()
    
    if rf is None or tfidf is None:
        print("[ERROR] Model or Vectorizer not loaded!")
        return "[ERROR] ML model is missing!", None

    text = extract_text_from_file(file)
    print("[DEBUG] Extracted Resume Text:", text,)  # ✅ Check if resume text is extracted

    if not text:
        return "[ERROR] Invalid or unsupported file format!", None

    try:
        text_vectorized = tfidf.transform([text]).toarray()
        print("Vectorized Input:", text_vectorized)  # ✅ Check if text is vectorized properly

        predicted_job = rf.predict(text_vectorized)[0]
        print("Predicted Job:", predicted_job)  # ✅ Check what job is predicted

        return None, predicted_job

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return "[ERROR] Prediction failed!", None


# Resume Se Text Extract Karne Ka Function
def extract_text_from_file(file):
    text = ""
    if file.filename.endswith(".pdf"):
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"[ERROR] PDF Processing Failed: {e}")
            return None
    elif file.filename.endswith(".docx"):
        try:
            doc = docx.Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            print(f"[ERROR] DOCX Processing Failed: {e}")
            return None
    else:
        print("[ERROR] Unsupported file format!")
        return None  # Unsupported file format

    return text.strip() if text.strip() else None  # No extractable text




app = Flask(__name__, template_folder="templates")

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_job = None
    error_message = None

    if request.method == "POST":
        if "resume" not in request.files:
            error_message = "No file uploaded!"
        else:
            file = request.files["resume"]
            if file.filename == "":
                error_message = "No selected file!"
            else:
                error_message, predicted_job = process_resume(file)  # Function Call

    return render_template("index.html", predicted_job=predicted_job or "", error_message=error_message or "")

if __name__ == "__main__":
    app.run(port= 5001, debug=True)

import fitz
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
import os
from flask import Flask, request, jsonify
import tempfile
from flask_cors import CORS

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_AI_API_KEY"))

app = Flask(__name__)
CORS(app)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = ""
    for page in doc:
        text = page.get_text()
        all_text += text + "\n"
    doc.close()  
    return all_text

def summarize_with_openai(text, word_count):
    """Summarize text using OpenAI GPT"""
    if word_count < 500:
        prompt = f"Summarize the following text in 3-5 clear bullet points:\n\n{text}"
    else:
        prompt = f"Summarize the following text in 5-7 clear bullet points:\n\n{text}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating summary with OpenAI: {str(e)}"

def summarize_with_gemini(text, word_count):
    """Summarize text using Google Gemini"""
    if word_count < 500:
        prompt = f"Summarize the following text in 3-5 clear bullet points:\n\n{text}"
    else:
        prompt = f"Summarize the following text in 5-7 clear bullet points:\n\n{text}"
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=800,
                candidate_count=1,
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"Error generating summary with Gemini: {str(e)}"

def summarize(pdf_path, ai_provider="openai"):
    text = extract_text_from_pdf(pdf_path)
    
    # Check if text was extracted
    if not text.strip():
        return "No text could be extracted from the PDF."
    
    
    word_count = len(text.split())
    
    # Route to appropriate AI provider
    if ai_provider == "openai":
        return summarize_with_openai(text, word_count)
    elif ai_provider == "gemini":
        return summarize_with_gemini(text, word_count)
    else:
        return "Invalid AI provider selected."

@app.route("/summarize", methods=["POST"])
def summarize_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "File is not a PDF"}), 400

        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file.save(tmp.name)
            summary = summarize(tmp.name)
            
        # Clean up temporary file
        os.unlink(tmp.name)
        
        return jsonify({"summary": summary})
    
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":    
    app.run(debug=True, port=5000, host='0.0.0.0')
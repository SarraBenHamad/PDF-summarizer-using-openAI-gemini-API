# PDF Summarizer

A simple web app that lets you upload a PDF, summarizes its content using OpenAI's GPT or Gemini you get to choose, and allows you to download the summary.

---

## Features

- Upload PDF files via a web interface.
- choose OpenAI or Gemini models
- Summarizes PDF content into 3-5 bullet points using OpenAI's GPT-3.5 Turbo.
- Displays the summary on the page.
- Download the summary as a `.txt` file.

---

## Requirements

- Python 3.7+
- OpenAI API key
- Flask
- Flask-CORS
- PyMuPDF (fitz)
- python-dotenv

---

## Setup

1. Clone the repository or copy the project files.

2. Install dependencies:

```bash
pip install flask flask-cors pymupdf python-dotenv openai

# Statistics RAG Assistant

A Streamlit app that answers questions using information retrieved from a Statistics PDF. It extracts and chunks the PDF, creates embeddings with FastEmbed, stores them in an in-memory Qdrant collection, and generates context-grounded answers with NVIDIA Nemotron.

## Features

- Upload a PDF or use the included `STATS.pdf`.
- Retrieve up to three relevant text chunks for each question.
- Generate answers with the NVIDIA API.
- Keep the vector collection in memory for the current app session.

## Requirements

- Python 3.10 or later
- An NVIDIA API key

## Setup

In PowerShell, from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r Requirement.txt
```

Create a `.env` file in the project directory and add your NVIDIA API key:

```text
NVIDIA_API_KEY=your_nvidia_api_key
```

Keep `.env` private; it is excluded from Git.

## Run

```powershell
streamlit run Main.py
```

Open the local URL printed by Streamlit. Upload a PDF, or leave the uploader empty to use `STATS.pdf`. Enter a question and select **Get Answer**.

The first request may take longer because the embedding model may need to be downloaded. The in-memory vector database is rebuilt when the app process restarts.

## Notes

- Answers are instructed to use the retrieved PDF context and to say when the answer is not in that context.
- Scanned/image-only PDFs may need OCR before their content can be searched.
- `Requirement.txt` contains dependencies for this and related project functionality.

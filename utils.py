from pypdf import PdfReader
import requests
from bs4 import BeautifulSoup

def process_pdf(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def process_text(file):
    return file.read().decode("utf-8")

def process_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        return " ".join(t.strip() for t in soup.stripped_strings)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error fetching URL: {e}")

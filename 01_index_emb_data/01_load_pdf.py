import os
import fitz
import re
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, pdf_path, output_dir, max_chunk_length=512):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.max_chunk_length = max_chunk_length

    def open_pdf(self):
        try:
            if not os.path.exists(self.pdf_path):
                raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
            return fitz.open(self.pdf_path)
        except Exception as e:
            logger.error(f"Failed to open PDF file: {str(e)}")
            raise

    def extract_text_from_pdf(self, pdf_document):
        text = ""
        for page_num in tqdm(range(len(pdf_document)), desc="Extracting text from PDF", unit="page"):
            page = pdf_document[page_num]
            text += page.get_text("text") + "\n\n"
        return text

    @staticmethod
    def clean_text(text):
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\t+', ' ', text)
        text = re.sub(r'\r+', '', text)
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'(\n\s*\d+\s*\n)', '', text)
        return text.strip()

    def split_text_into_chunks(self, text):
        sentences = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > self.max_chunk_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def save_chunks(self, chunks):
        os.makedirs(self.output_dir, exist_ok=True)

        for i, chunk in enumerate(chunks):
            output_path = os.path.join(self.output_dir, f"chunk_{i+1}.txt")
            with open(output_path, "w", encoding="utf-8") as file:
                file.write(chunk)
            logger.info(f"Chunk {i+1} saved to: {output_path}")

    def process(self):
        try:
            pdf_document = self.open_pdf()
            text = self.extract_text_from_pdf(pdf_document)
            cleaned_text = self.clean_text(text)
            text_chunks = self.split_text_into_chunks(cleaned_text)
            self.save_chunks(text_chunks)
            logger.info(f"Successfully processed {self.pdf_path} and saved {len(text_chunks)} chunks.")
        except Exception as e:
            logger.error(f"An error occurred during the PDF processing: {str(e)}")
            raise

def main():
   
    current_dir = os.path.dirname(os.path.abspath(__file__))

    pdf_path = os.path.join(current_dir, "..", "00_data", "doc.pdf")
    output_dir = os.path.join(current_dir, "..", "00_data", "chunks")
    
    max_chunk_length = 512

    processor = PDFProcessor(
        pdf_path,
        output_dir,
        max_chunk_length=max_chunk_length
    )

    try:
        processor.process()
    except Exception as e:
        logger.error(f"Failed to process PDF: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()

FROM python:3.9

WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


RUN pip install streamlit

WORKDIR /app/03_stand_alone_llm

CMD ["streamlit", "run", "custom_prompt_ollama.py"]

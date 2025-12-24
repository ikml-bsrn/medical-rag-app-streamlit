FROM python:3.13.5-slim

WORKDIR /app

ENV STREAMLIT_BROWSER_GATHER_USAGE_STAT=false

# Copy necessary files to /app
COPY main.py .
COPY rag_retrieve.py .
COPY requirements.txt .
COPY medquad_chroma_db_2 ./medquad_chroma_db_2

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "main.py", "--server.port=7860", "--server.address=0.0.0.0"]


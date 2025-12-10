FROM python:3.13.5-slim

WORKDIR /app

ENV STREAMLIT_BROWSER_GATHER_USAGE_STAT=false

# Copy necessary files to /app
COPY app.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["python", "app.py"]


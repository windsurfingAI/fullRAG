FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ui.py .

EXPOSE 8501

# Configuration Streamlit pour Docker (désactivation du CORS et binding global)
CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
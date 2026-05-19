# Use the official Python image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy the rest of the application
COPY . .

RUN chown -R appuser:appuser /app
USER appuser

# Expose the port Streamlit runs on
EXPOSE 8080

# Command to run the application
# Use port 8080 for Cloud Run
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]

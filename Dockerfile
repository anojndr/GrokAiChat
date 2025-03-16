FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# Use uvicorn instead of the Flask development server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
FROM python:3.12.9-slim

WORKDIR /app

RUN pip install oceanbench==0.0.2

COPY main.py .

CMD ["python", "main.py"]

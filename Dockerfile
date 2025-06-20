FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

ENV POLARS_MAX_THREADS=$((nproc))
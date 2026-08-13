FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOMEVALUE_DATA_ROOT=/app/data/processed

WORKDIR /app

RUN addgroup --system homevalue && \
    adduser --system --ingroup homevalue --home /home/homevalue \
      --shell /bin/bash homevalue && \
    install -d -m 0700 -o homevalue -g homevalue /home/homevalue/.ssh

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=homevalue:homevalue . .

USER homevalue
EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

FROM python:3.9-slim-buster as wheel_builder

COPY ./requirements.txt /requirements.txt

RUN apt-get update \
    && apt-get install cmake --no-install-recommends -y \
    && pip install --upgrade pip \
    && pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt \
    && apt-get purge cmake -y \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

FROM python:3.9-slim-buster as production

COPY --from=wheel_builder /wheels /wheels

RUN pip install --upgrade pip \
    && pip install --no-cache /wheels/* \
    && rm -rf /wheels

ENV PYTHONUNBUFFERED=1

WORKDIR app
COPY . /app

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

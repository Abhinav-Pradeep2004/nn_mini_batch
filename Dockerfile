FROM python:3.10.14-bookworm

RUN pip install --upgrade pip

COPY src /app/src

COPY auto.sh /app/auto.sh

WORKDIR /app

RUN mkdir -p /app/src/trained_models

RUN chmod -R 777 /app/src

RUN chmod +x /app/auto.sh

RUN pip install -r /app/src/requirements.txt

ENV PYTHONPATH=${PYTHONPATH}:/app/src

ENTRYPOINT ["/app/auto.sh"]
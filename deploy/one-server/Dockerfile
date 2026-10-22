# One image for MLflow, the application and the jobs of the one-server stack.
# Author: Yves Schillings.
FROM python:3.12-slim

WORKDIR /app
COPY deploy/one-server/requirements.txt deploy/one-server/requirements.txt
RUN pip install --no-cache-dir -r deploy/one-server/requirements.txt
COPY examples examples
COPY deploy/one-server/app deploy/one-server/app

ENV PYTHONPATH=/app/examples \
    PYTHONUNBUFFERED=1
# Not root: the processes run as an unprivileged user that owns only its own folders.
RUN useradd --uid 10001 --create-home runner \
    && mkdir -p /app/results /approvals /mlartifacts \
    && chown -R runner /app/results /approvals /mlartifacts
USER runner

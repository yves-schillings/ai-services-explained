#!/bin/bash
# Two databases with two separate users: MLflow cannot read the retrieval data, and the
# application's retrieval user cannot read MLflow. Author: Yves Schillings.
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-SQL
  CREATE USER mlflow WITH PASSWORD '${MLFLOW_DB_PASSWORD}';
  CREATE DATABASE mlflow OWNER mlflow;
  CREATE USER rag WITH PASSWORD '${RAG_DB_PASSWORD}';
  CREATE DATABASE rag OWNER rag;
  REVOKE ALL ON DATABASE mlflow FROM PUBLIC;
  REVOKE ALL ON DATABASE rag FROM PUBLIC;
SQL
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname rag -c "CREATE EXTENSION IF NOT EXISTS vector;"

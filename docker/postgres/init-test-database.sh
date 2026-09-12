#!/bin/sh
set -eu

psql \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" \
    --set=test_database="$POSTGRES_TEST_DB" \
    --set=ON_ERROR_STOP=1 <<'SQL'
SELECT format('CREATE DATABASE %I', :'test_database')
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = :'test_database'
)\gexec
SQL

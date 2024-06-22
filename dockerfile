#INSTRUCTION指令    arugments操作
FROM python:3.10-slim as python

WORKDIR /the/workdir/app

ENV MY_ENV_VAR="Hello world"

COPY test.txt .

RUN mkdir -p new

COPY test.txt new/

CMD echo "my environment variable is :${MY_ENV_VAR}"
ARG BASE_CONTAINER=python:latest
FROM $BASE_CONTAINER

WORKDIR /usr/src/app

COPY . .
USER root
RUN pip install -r requirements.txt

CMD ["python", "app.py"]
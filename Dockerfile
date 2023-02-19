FROM python:3.11-slim-buster

WORKDIR /code
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY . .

ENV PYTHONPATH /code

EXPOSE 5000
ENTRYPOINT ["python3"]
CMD ["main.py"]

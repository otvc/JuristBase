FROM python:3.10.6

WORKDIR /usr/src/juristapp

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "./backend/engine.py"]

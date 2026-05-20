FROM python:3.11.9

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]

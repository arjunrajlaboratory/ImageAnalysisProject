FROM python:3.12-slim

RUN pip install --no-cache-dir flask numpy tifffile

WORKDIR /app
COPY server.py .

EXPOSE 5555

CMD ["python", "server.py", "--images-dir", "/images", "--port", "5555", "--output-dir", "/output"]

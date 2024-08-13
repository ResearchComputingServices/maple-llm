# maple-llm
Large Language Model backend for Maple project

# Running the server
```
gunicorn --certfile cert.pem --keyfile key.pem -b 0.0.0.0:5000 --threads 8 --timeout 0  gptserver:app
```

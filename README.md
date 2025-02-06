# maple-llm
Large Language Model backend for Maple project

# Install
```bash
sudo ./install.sh
```
# Running the server
```
gunicorn --certfile /etc/nginx/aimodel_cert.pem --keyfile /etc/nginx/aimodel_key.pem -b 0.0.0.0:5000 --threads 8 --timeout 0  gptserver:app
```


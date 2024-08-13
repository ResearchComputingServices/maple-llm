import logging
from gpt4all import GPT4All
import timeit
from flask import Flask, abort, request, jsonify
from threading import Lock
from dotenv import load_dotenv
import os
from flask_cors import CORS, cross_origin
from models import MLModel, ModelType
from functools import wraps
import coloredlogs
# from flask_lt import run_with_lt

coloredlogs.install(level='DEBUG')
# logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


config = load_dotenv('production.env')
API_KEY = os.getenv('API_KEY', None)

models = MLModel(load_default=True)

app = Flask(__name__)
cors=CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
# run_with_lt(app)

def require_api_key(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        api_key = request.headers.get('x-api-key')
        if api_key and api_key == API_KEY:
            return f(*args, **kwargs)
        else:
            abort(401)
    return decorator

@app.route('/llm/generate', methods=['POST'])
@cross_origin()
@require_api_key
def generate():
    data = request.json
    log.debug(f"Got request: {data}")
    required_fields = ['model_type', 'prompt']
    optional_fields = ['max_tokens']
    if not all(field in data for field in required_fields):
        missing_fiels = [field for field in required_fields if field not in data]
        abort(400, f"Missing fields: {missing_fiels}")
    if 'max_tokens' not in data:
        data['max_tokens'] = 512
    try:
        response = models.answer(
            data['model_type'],
            data['prompt'],
            max_tokens=data['max_tokens']
            )
        return jsonify(response)
    
    except Exception as e:
        log.error(f"Error generating response: {e}")
        abort(500)
    
    

@app.route('/llm/capabilities', methods=['GET'])
@require_api_key
def capabilities():
    log.debug("Requested capabilities")
    response = jsonify({
        'models': models.model_names,
        'endpoints': {
            'generate': [
                'prompt', 
                'max_tokens']
            }
    })
    # response.headers.add('Access-Control-Allow-Origin', '*')
    log.debug(response)
    return response

@app.route('/', methods=['GET'])
def main_page():
    return jsonify("GPT4All server is running")

# gunicorn --certfile cert.pem --keyfile key.pem -b 0.0.0.0:5000 -w 8 --timeout 0  gptserver:app

if __name__ == '__main__':
    app.run( 
        port=5000,
        # ssl_context=('cert.pem', 'key.pem'),
        ssl_context='adhoc',
        # debug=True,
        )
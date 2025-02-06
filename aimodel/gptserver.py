import logging
from gpt4all import GPT4All
import timeit
from flask import Flask, abort, request, jsonify, send_from_directory
from threading import Lock
from dotenv import load_dotenv
import os
from flask_cors import CORS, cross_origin
from .models import MLModel, ModelType
from functools import wraps

log = logging.getLogger(__name__)

API_KEY = os.getenv('API_KEY', None)

app = Flask(__name__)

def require_api_key(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        if API_KEY is not None:
            api_key = request.headers.get('x-api-key')
            if api_key and api_key == API_KEY:
                return f(*args, **kwargs)
            else:
                abort(401)
        else:
            return f(*args, **kwargs)
    return decorator

@cross_origin()
@require_api_key
def generate():
    """
    Generates a response using a model based on the provided json data with fields 'model_type', 'prompt', and 'max_tokens'.
    Returns:
    - A JSON response containing the generated response.
    Raises:
    - 400 error if any required fields are missing in the data.
    - 500 error if there is an error generating the response.
    """

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
        response = app.config['models'].answer(
            data['model_type'],
            data['prompt'],
            max_tokens=data['max_tokens']
            )
        return jsonify(response)
    
    except Exception as e:
        log.error(f"Error generating response: {e}")
        abort(500)
    
    
@require_api_key
def capabilities():
    """
    Returns the capabilities of the server.
    :return: A JSON response containing the available models and endpoints.
    """
    
    log.debug("Requested capabilities")
    response = jsonify({
        'models': app.config['models'].model_names,
        'endpoints': {
            'generate': [
                'prompt', 
                'max_tokens'],
            
        "article_summary": [
            'articles',
            'max_tokens',
        ]
            },
    })
    log.debug(response)
    return response

@require_api_key
def article_summary(*args, **kwargs):
    """
    Generates a summary of the provided articles.
    Returns:
    - A JSON response containing the generated summary.
    Raises:
    - 400 error if any required fields are missing in the data.
    - 500 error if there is an error generating the summary.
    """
    
    data = request.json
    log.debug(f"Got request: {data}")
    required_fields = ['articles', 'model_type', 'prompt']
    optional_fields = ['max_tokens']
    if not all(field in data for field in required_fields):
        missing_fiels = [field for field in required_fields if field not in data]
        abort(400, f"Missing fields: {missing_fiels}")
    if 'max_tokens' not in data:
        data['max_tokens'] = 512
    try:
        response = app.config['models'].article_summary(
            model_type=data['model_type'],
            articles=data['articles'],
            max_tokens=data['max_tokens'],
            prompt=data['prompt']
            )
        log.debug(f"Response for summary_article: {response}")
        return jsonify(response)
    
    except Exception as e:
        log.error(f"Error generating response: {e}")
        abort(500)

@require_api_key
def topic_name():
    """
    Generates a topic name given a list of keywords.
    Returns:
    - A JSON response containing the generated summary.
    Raises:
    - 400 error if any required fields are missing in the data.
    - 500 error if there is an error generating the summary.
    """
    
    data = request.json
    log.debug(f"Got request: {data}")
    required_fields = ['keywords', 'model_type', 'prompt']
    
    if not all(field in data for field in required_fields):
        missing_fiels = [field for field in required_fields if field not in data]
        abort(400, f"Missing fields: {missing_fiels}")
    
    try:
        response = app.config['models'].topic_name(
            model_type=data['model_type'],
            keywords=data['keywords'],
            prompt=data['prompt']
            )
        return jsonify(response)
    
    except Exception as e:
        log.error(f"Error generating response: {e}")
        abort(500)


@require_api_key
def bullet_point():
    """
    Generates a bullet point summary for given a list of articles.
    Returns:
    - A JSON response containing the generated summaries.
    Raises:
    - 400 error if any required fields are missing in the data.
    - 500 error if there is an error generating the summary.
    """
    
    data = request.json
    log.debug(f"Got request: {data}")
    required_fields = ['articles', 'model_type', 'prompt']
    
    if not all(field in data for field in required_fields):
        missing_fiels = [field for field in required_fields if field not in data]
        abort(400, f"Missing fields: {missing_fiels}")
    if 'max_tokens' not in data:
        data['max_tokens'] = 512
    
    try:
        response = app.config['models'].bullet_point_summary(
            model_type=data['model_type'],
            articles=data['articles'],
            prompt=data['prompt'],
            max_tokens=data['max_tokens']
            )
        return jsonify(response)
    
    except Exception as e:
        log.error(f"Error generating response: {e}")
        abort(500)
        
def favicon(): 
    log.debug("Requested favicon")
    return send_from_directory(
        os.path.join(
            app.root_path, 
            'static'), 
        'favicon.ico', 
        mimetype='image/vnd.microsoft.icon')

def main_page():
    """
    Returns a JSON response indicating that the server is running.
    :return: JSON response
    """
    return jsonify("Server is running!")


def register_urls(app):
    """
    Register the URLs for the application.
    Parameters:
    - app: The Flask application object.
    Returns:
    None
    """
    
    app.add_url_rule('/llm/generate', view_func=generate, methods=['POST'])
    app.add_url_rule('/llm/capabilities', view_func=capabilities, methods=['GET'])
    app.add_url_rule('/', view_func=main_page, methods=['GET'])
    app.add_url_rule('/llm', view_func=main_page, methods=['GET'])
    app.add_url_rule('/favicon.ico', view_func=favicon, methods=['GET'])
    app.add_url_rule('/llm/article_summary', view_func=article_summary, methods=['POST'])
    app.add_url_rule('/llm/topic_name', view_func=topic_name, methods=['POST'])
    app.add_url_rule('/llm/bullet_point', view_func=bullet_point, methods=['POST'])
    
def create_app(models: MLModel | None = None):
    log.debug("Creating app")
    log.debug(f"Models: {models}")
    if models is None:
        app.config['models'] = MLModel(
            load_default=True,
            mistral_path=os.getenv('MISTRAL_PATH', None),
            hugging_face_api_key=os.getenv('MISTRAL_API_KEY', None),
            )
            
    else:
        app.config['models'] = models
    
    cors=CORS(app)
    app.config['CORS_HEADERS'] = 'Content-Type'
    API_KEY = os.getenv('API_KEY', None)
    app.config['API_KEY'] = API_KEY

    register_urls(app)
    
    return app
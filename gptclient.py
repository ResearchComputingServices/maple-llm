from aiohttp_requests import requests
from aiohttp import BasicAuth
from requests.auth import HTTPBasicAuth
import asyncio
from dotenv import load_dotenv
import os

load_dotenv('production.env')
API_KEY = os.getenv('API_KEY', None)
VERIFY_SSL = os.getenv('VERIFY_SSL', 'False').lower() in [1, 'true', 'yes', 't']

NQUESTIONS=3

async def get_response(prompt, max_tokens):
    response = await requests.post(
        'https://127.0.0.1:5000/generate', 
        json={'prompt': prompt, 'max_tokens': max_tokens},
        headers={'x-api-key': API_KEY},
        ssl=VERIFY_SSL,
        timeout=None,
    )
    return response

async def main():
    responses = [get_response(f'How can I run LLMs efficiently on my laptop ({i})?', 1004*i+20) for i in range(NQUESTIONS)]
    res = await asyncio.gather(*responses)
    [print(await r.json()) for r in res]
    print(responses)

if __name__ == '__main__':
    
    asyncio.run(main())
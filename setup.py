from setuptools import setup

setup(
    name='aimodel',
    version='1.0',
    description='Python package for AI model',
    author='Research and Computing Services',
    author_email='its.rcs@carleton.ca',
    packages=['aimodel'],
    install_requires=[
        'numpy',
        'tensorflow',
        'torch',
        'gpt4all',
        'transformers',
        'mistral_inference',
        'sentencepiece',
        'flask',
        'flask-cors',
        'python-dotenv',
        'coloredlogs',
        'cryptography',
    ],
)

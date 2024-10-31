import logging
from dotenv import load_dotenv
import os
from aimodel import MLModel, create_app
import argparse
import coloredlogs

fmt = "%(asctime)s %(hostname)s %(name)s[%(process)d] %(funcName)s(%(lineno)s) %(levelname)s %(message)s"
    
argparser = argparse.ArgumentParser()
argparser.add_argument("-p", "--port", type=int, default=5000)
argparser.add_argument("-l", "--log-level", type=str, default=None)


load_dotenv("production.env")

def app():
    args = argparser.parse_known_args()[0]
    application = app_(**vars(args), fmt=fmt)
    logging.debug(f"args: {args}")
    return application

def app_(**kwargs):
    log_level = kwargs.get("log_level", "DEBUG")
    coloredlogs.install(log_level, **kwargs)
    logging.debug(f"kwargs: {kwargs}")
    os.environ["MISTRAL_PATH"] = "/data"
    os.environ["MISTRAL_API_KEY"]= os.getenv("HF_TOKEN", None)
    
    application = create_app()
    
    return application


if __name__ == "__main__":
    args = argparser.parse_args()
    coloredlogs.install(
        level=args.log_level,
        fmt=fmt,
    )
    application = app_(**vars(args), fmt=fmt)
    application.run(port=args.port, ssl_context="adhoc")

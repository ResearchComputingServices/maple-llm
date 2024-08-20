from dotenv import load_dotenv
import os
from aimodel import MLModel, create_app
import argparse
import coloredlogs


argparser = argparse.ArgumentParser()
argparser.add_argument("-p", "--port", type=int, default=5000)
argparser.add_argument("-l", "--log-level", type=str, default=None)


load_dotenv("production.env")


def app(**kwargs):
    print(kwargs)
    os.environ["MISTRAL_PATH"] = "/data"
    os.environ["MISTRAL_API_KEY"]= os.getenv("HF_TOKEN", None)
    
    app = create_app()
    log_level = kwargs.get("log_level", "INFO")
    coloredlogs.install(log_level)
    return app


if __name__ == "__main__":
    args = argparser.parse_args()
    coloredlogs.install(level=args.log_level)
    app = app(**vars(args))
    app.run(port=args.port, ssl_context="adhoc")

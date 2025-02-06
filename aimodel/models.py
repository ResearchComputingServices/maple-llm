from abc import abstractmethod
from functools import wraps
from typing import Dict
import json
import logging
from pathlib import Path
from threading import Lock
import timeit
from typing import Protocol
from gpt4all import GPT4All
from huggingface_hub import snapshot_download, login
from transformers import AutoModelForCausalLM, AutoTokenizer
from enum import Enum
import os
import torch
import re
import gc

log = logging.getLogger("LLM")


class ModelType(Enum):
    """
    Enum class representing default model types.

    Attributes:
        GPT4ALL (str): Represents the GPT4ALL model type.
        MISTRAL (str): Represents the MISTRAL model type.
    """

    GPT4ALL = "gpt4all"
    MISTRAL = "mistral"


class Answer(Protocol):
    """
    Protocol defining the interface for the Answer class.
    Methods:
    - __call__(model: object, prompts: list[str], max_tokens: int) -> list[str]:
        Generates a list of answers based on the given model, prompts, and maximum number of tokens.
    Parameters:
    - model (object): The model to use for generating answers.
    - prompts (list[str]): A list of prompts to generate answers for.
    - max_tokens (int): The maximum number of tokens to generate for each answer.
    Returns:
    - list[str]: A list of generated answers.
    """

    def __call__(
        self, model: object, prompts: list[str], max_tokens: int
    ) -> list[str]: ...


class ArticleSummary(Protocol):
    """
    Protocol for generating summaries using a model.
    Args:
        model (object): The model used for generating summaries.
        articles (list[str]): List of articles to generate summaries for.
        max_tokens (int): The maximum number of tokens in the generated summary.
    Returns:
        list[str]: List of generated summaries.
    """
    
    def __call__(self, model: object, articles: list[str], max_tokens: int, prompt: str|None) -> list[str]: ...


class TopicName(Protocol):
    """
    Protocol for a callable object that takes a model and a list of keywords as input and returns a list of strings.
    Args:
        model (object): The model object.
        keywords (list[str]): A list of keywords.
    Returns:
        list[str]: A list of strings.
    """
    
    def __call__(self, model: object, keywords: list[str], prompt: str|None) -> str: ...


class BulletPointSummary(Protocol):
    """
    Protocol for generating bullet point summaries using a model.
    Args:
        model (object): The model used for generating summaries.
        articles (list[str]): List of articles to generate summaries for.
        max_tokens (int): The maximum number of tokens in the generated summary.
    Returns:
        list[str]: List of generated summaries.
    """
    
    def __call__(self, model: object, articles: list[str], max_tokens: int, prompt: str|None) -> list[str]: ...
    
    
class LoadMLModel(Protocol):
    """
    Protocol for loading a machine learning model.
    The `LoadMLModel` protocol defines the interface for a callable object that loads a machine learning model.
    The `__call__` method should be implemented to return the loaded model.
    Returns:
        object: The loaded machine learning model.
    """

    def __call__(self) -> object: ...
    
class UnloadMLModel(Protocol):
    """
    Protocol for unloading a machine learning model.
    The `UnloadMLModel` protocol defines the interface for a callable object that unloads a machine learning model.
    The `__call__` method should be implemented to free up memory for a given model.
    """

    def __call__(self) -> None: ...


    
class BaseModel:
    def create_summary_prompts(f):
        DEFAULT_SUMMARY_ARTICLE_KEYWORD = '**articles**'
        @wraps(f)
        def wrapper(*args, **kwargs):
            if 'prompt' in kwargs and 'articles' in kwargs:
                prompts = []
                for article in kwargs['articles']:
                    if DEFAULT_SUMMARY_ARTICLE_KEYWORD in kwargs['prompt']:
                        prompts.append(kwargs['prompt'].replace(DEFAULT_SUMMARY_ARTICLE_KEYWORD, article))
                    else:
                        prompts.append(kwargs['prompt'] + article)
            return f(*args, **kwargs, prompts=prompts)
        return wrapper

    def create_bullet_point_prompt(f):
        DEFAULT_BULLET_POINT_ARTICLE_KEYWORD = '**articles**'
        @wraps(f)
        def wrapper(*args, **kwargs):
            if 'prompt' in kwargs and 'articles' in kwargs:
                if DEFAULT_BULLET_POINT_ARTICLE_KEYWORD in kwargs['prompt']:
                    prompt = kwargs['prompt'].replace(DEFAULT_BULLET_POINT_ARTICLE_KEYWORD, json.dumps(kwargs['articles'], indent=2 ))
                else:
                    prompt = kwargs['prompt'] + '\n' + json.dumps(kwargs['articles'], indent=2 )
            return f(*args, **kwargs, final_prompt=prompt)
        return wrapper
    
    def create_topic_name_prompt(f):
        DEFAULT_TOPIC_NAME_KEYWORD = '**keywords**'
        @wraps(f)
        def wrapper(*args, **kwargs):
            if 'prompt' in kwargs and 'keywords' in kwargs:
                if DEFAULT_TOPIC_NAME_KEYWORD in kwargs['prompt']:
                    final_prompt = kwargs['prompt'].replace(DEFAULT_TOPIC_NAME_KEYWORD, json.dumps(kwargs['keywords']))
                else:
                    final_prompt = kwargs['prompt'] + '\n' + json.dumps(kwargs['keywords'])
            return f(*args, **kwargs, final_prompt=final_prompt)
        return wrapper
    
    @abstractmethod
    def load_model(self) -> object:
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def unload_model(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def answer(self, model: object, prompts: list[str], max_tokens: int) -> list[str]: 
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def artile_summary(self, model: object, articles: list[str], max_tokens: int, prompt: str, **kwargs) -> list[str]: 
        """
        Generate a summary for a list of articles using a given model.
        Args:
            model (object): The model used for generating the summary.
            articles (list[str]): A list of articles to summarize.
            max_tokens (int): The maximum number of tokens in the generated summary.
            prompt (str): The prompt to use for generating the summary.
            prompts (list[str], optional): A list of prompts to use for generating the summary. Defaults to None.
        Returns:
            list[str]: A list of summaries for the given articles.
        Raises:
            NotImplementedError: If the method is not implemented by subclasses.
        """
        
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def topic_name(self, model: object, keywords: list[str], prompt: str, **kwargs) -> str: 
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def bullet_point_summary(self, model: object, articles: list[str], max_tokens: int, prompt: str, **kwargs) -> list[str]: 
        raise NotImplementedError("Subclasses must implement this method")


class GPT4AllModel(BaseModel):
    """
    A class representing a GPT-4 model for generating responses to prompts.
    Attributes:
        lock (Lock): A lock object for thread safety.
        model (object): The GPT-4 model instance.
    """

    def __init__(self) -> None:
        self.lock = Lock()
        self.model = None
        self._device = "kompute" if torch.cuda.is_available() else "cpu"

    def load_model(self) -> object:
        """
        Loads and returns the model if it exists, otherwise creates a new instance of GPT4All model and returns it.
        Returns:
            object: The loaded or newly created GPT4All model instance.
        """

        with self.lock:
            if self.model is not None:
                return self.model
            log.info(f"Loading GPT4All model")
            self.model = GPT4All(
                'GPT4All-Community/Meta-Llama-3.1-8B-Instruct-128k-GGUF', #"Meta-Llama-3-8B-Instruct.Q4_0.gguf",
                device=self._device,
                n_ctx=8192, #4096
            )
            return self.model

    def unload_model(self) -> None:
        """
        Unloads the model and frees up memory.
        """

        with self.lock:
            del self.model
            self.model = None
                    
    def answer(self, model: object, prompts: list[str], max_tokens: int) -> list[str]:
    
        """
        Generates a response using the given model, prompts, and maximum number of tokens.
        Args:
            model (object): The model to use for generating the response.
            prompts (list[str]): A list of prompts to generate the response from.
            max_tokens (int): The maximum number of tokens to generate in the response.
        Returns:
            list[str]: A list of generated responses for each prompt.
        """

        response = []
        with self.lock:
            with model.chat_session():
                for p_i, prompt in enumerate(prompts):
                    response.append(
                        model.generate(
                            prompt, 
                            max_tokens=max_tokens,
                            )
                        )
        return response

    @BaseModel.create_summary_prompts
    def artile_summary(
        self, 
        model: GPT4All,  
        articles: list[str], 
        max_tokens: int,
        prompt: str,
        **kwargs
        ) -> list[str]:
        log.debug(f"Generating article summaries using GPT4All...")
        if 'prompts' in kwargs:
            log.debug(f"Prompts: {kwargs['prompts']}")
        response = []
        with self.lock:
            if 'prompts' in kwargs:
                prompts = kwargs['prompts']
                for prompt in prompts:
                    response.append(
                        model.generate(
                            prompt, 
                            max_tokens=max_tokens,
                            )
                        )
            else:
                for article in articles:
                    article = re.sub(r'[^\w\s,]', '', article)
                    if "**articles**" in prompt:
                        prompt_ = prompt.replace("**articles**", article)
                    else:
                        prompt_ = prompt + '\n'.join(article)
                    
                    response.append(
                        model.generate(
                            prompt_, 
                            max_tokens=max_tokens,
                            )
                    )
        return response
    
    @BaseModel.create_topic_name_prompt
    def topic_name(self, model: object, keywords: list[str], prompt: str, **kwargs) -> str:
        if 'final_prompt' in kwargs:
            final_prompt = kwargs['final_prompt']
        else:
            final_prompt = prompt + '\n' + json.dumps(keywords)
        with self.lock:
            response = model.generate(
                final_prompt, 
                max_tokens=100)
        response = response.split('<|eot_id|>')[0]
        return response
    
    @BaseModel.create_bullet_point_prompt
    def bullet_point_summary(self, model: object, articles: list[str], max_tokens: int, prompt: str, **kwargs) -> list[str]:
        if 'final_prompt' in kwargs:
            final_prompt = kwargs['final_prompt']
        else:
            final_prompt = prompt + '\n' + json.dumps(articles, indent=2)
        
        log.debug(f"Generating bullet point summary using GPT4All...\n{final_prompt}")
        with self.lock:
            response = model.generate(
                final_prompt,
                max_tokens=max_tokens,
                )
        log.debug(f"Generated bullet point summary: {response}")
        return response

class MistralModel(BaseModel):
    """
    A class representing the Mistral Model.
    Attributes:
        lock (Lock): A lock object for thread safety.
        model (object): The loaded Mistral model.
    """

    def __init__(self, mistral_path: str | None, hugging_face_api_key: str | None) -> None:
        self.lock = Lock()
        # model_id = "mistralai/Mistral-7B-v0.3"
        if hugging_face_api_key is not None:
            login(token=hugging_face_api_key)
        self._model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        self._pre_download(mistral_path, model_id=self._model_id)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
    
    def _pre_download(
        self, 
        mistral_folder: str | None = None,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        log.debug(f"Downloading mistral model and storing in {mistral_folder}")
        if mistral_folder is not None:
            mistral_models_path = Path(mistral_folder).joinpath('mistral_models', model_id)
        else:
            mistral_models_path = Path.home().joinpath('mistral_models', model_id)
            
        mistral_models_path.mkdir(parents=True, exist_ok=True)
        log.debug(f"Downloading model to {mistral_models_path}")
        snapshot_download(
            repo_id=model_id, 
            allow_patterns=[
                "params.json", 
                "consolidated.safetensors", 
                "tokenizer.model.v3"], 
            local_dir=mistral_models_path,
            cache_dir=mistral_models_path,
            )

    def load_model(self) -> object:
        with self.lock:
            if self.model is not None:
                return self.model
            log.info(f"Loading Mistral model")
            self.model = AutoModelForCausalLM.from_pretrained(self._model_id)
            self.model.to(self._device)
            self.tokenizer = AutoTokenizer.from_pretrained(self._model_id)
            return self.model
    
    def unload_model(self) -> None:
        """
        Unloads the model and frees up memory.
        """
        with self.lock:
            try:
                if self._device != "cpu":
                    self.model.to("cpu")
            except:
                log.warning(f"Could not move model to cpu")
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
                    
    def answer(self, model: object, prompts: list[str], max_tokens: int) -> list[str]:
        response = []
        with self.lock:
            for p_i, prompt in enumerate(prompts):
                t_start = timeit.default_timer()
                inputs = self.tokenizer(prompt, return_tensors="pt")
                t_tokenized = timeit.default_timer() - t_start
                log.debug(f"Time to tokenize: {t_tokenized}")
                inputs.to(self._device)
                outputs= model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens,
                    repetition_penalty=1.15,
                    )
                t_generate = timeit.default_timer() - t_tokenized
                log.debug(f"Time to generate: {t_generate}")
                model_response = self.tokenizer.decode(
                    outputs[0][len(inputs['input_ids'][0]):], 
                    skip_special_tokens=True)
                log.debug(f"Time to decode: {timeit.default_timer() - t_generate}")
                log.debug(f"Generated output ({timeit.default_timer() - t_start}s): {model_response}")
                
                response.append(
                    model_response
                    )
        return response

    @BaseModel.create_summary_prompts
    def artile_summary(self, model: object, articles: list[str], max_tokens: int, prompt: str, **kwargs) -> list[str]:
        if 'prompts' in kwargs:
            prompts = kwargs['prompts']
        else:
            prompts = []
            for article in articles:
                prompts.append(prompt + article)
        response = self.answer(
            model=model,
            prompts=prompts, 
            max_tokens=max_tokens)
        return response

    @BaseModel.create_topic_name_prompt
    def topic_name(self, model: object, keywords: list[str], prompt: str, **kwargs) -> str:
        if 'final_prompt' in kwargs:
            final_prompt = kwargs['final_prompt']
        else:
            final_prompt = prompt + '\n' + json.dumps(keywords)
        response = self.answer(
            model=model,
            prompts=[final_prompt], 
            max_tokens=100)
        return response[0]
    
    @BaseModel.create_bullet_point_prompt
    def bullet_point_summary(self, model: object, articles: list[str], max_tokens: int, prompt: str, **kwargs) -> list[str]:
        if 'final_prompt' in kwargs:
            final_prompt = kwargs['final_prompt']
        else:
            final_prompt = prompt + '\n' + json.dumps(articles, indent=2)
        
        response = self.answer(
            model=model,
            prompts=[final_prompt],
            max_tokens=max_tokens,
        )
        # if numbered points
        bullet_points = re.split(r'\s(?=\d[\.\-\)\#])', response[0])
        
        if len(bullet_points) == 1:
            bullet_points = re.split(r'\s(?=[\*\-])', response[0])
            
        return bullet_points


class MLModel:
    """
    Represents a machine learning model.
    Args:
        load_default (bool, optional): Whether to load the default model. Defaults to True.
    Attributes:
        model: The loaded machine learning model.
    """

    def __init__(
        self, 
        load_default: bool = True,
        mistral_path: str | None = None,
        hugging_face_api_key: str | None = None) -> None:
        
        log.info(f"Initializing MLModel")
        self._models: Dict[str, BaseModel] = {}
        self._loaded = None
        self.model = None
        self._mlmodel_lock = Lock()
        if load_default:
            self._gpt4all = GPT4AllModel()
            self.add_model(
                model_name=ModelType.GPT4ALL.value, 
                load_model=self._gpt4all.load_model, 
                unload_model=self._gpt4all.unload_model,
                answer=self._gpt4all.answer,
                article_summary=self._gpt4all.artile_summary,
                topic_name=self._gpt4all.topic_name,
                bullet_point_summary=self._gpt4all.bullet_point_summary,
            )
            
            # TODO add mistral model
            if os.path.exists("/data"): 
                mistral_folder = "/data"
            else:
                mistral_folder = None
                
            self._mistral = MistralModel(mistral_path = mistral_path, hugging_face_api_key = hugging_face_api_key)
            self.add_model(
                model_name=ModelType.MISTRAL.value, 
                load_model=self._mistral.load_model, 
                unload_model=self._mistral.unload_model,
                answer=self._mistral.answer,
                article_summary=self._mistral.artile_summary,
                topic_name=self._mistral.topic_name,
                bullet_point_summary=self._mistral.bullet_point_summary,
                )

    def add_model(
        self, 
        model_name: str, 
        load_model: LoadMLModel, 
        unload_model: UnloadMLModel,
        answer: Answer,
        article_summary: ArticleSummary,
        topic_name: TopicName,
        bullet_point_summary: BulletPointSummary,
        ):
        """
        Adds a new model to the collection of models.
        Parameters:
        - model_name (str): The name of the model (should be unique).
        - load_model (LoadMLModel): The function to load the machine learning model.
        - answer (Answer): The answer associated with the model.
        Raises:
        - ValueError: If the model with the given name already exists.
        Returns:
        - None
        """
        if model_name in self._models:
            raise ValueError(f"Model {model_name} already exists")
        self._models[model_name] = {
            "name": model_name,
            "load_model": load_model,
            "unload_model": unload_model,
            "answer": answer,
            "article_summary": article_summary,
            "topic_name": topic_name,
            "bullet_point_summary": bullet_point_summary,
        }

    @property
    def model_names(self):
        """
        Returns a list of model names.
        Returns:
            list: A list of model names.
        """
        return list(self._models.keys())

    def _load_model(self, model_name: str = ModelType.GPT4ALL.value):
        """
        Load a specified model.
        Args:
            model_name (str, optional): The name of the model to load. Defaults to ModelType.GPT4ALL.value.
        Raises:
            ValueError: If the specified model name is not recognized.
        Returns:
            None
        """
        with self._mlmodel_lock:
            if model_name not in self._models:
                raise ValueError(f"Model {model_name} not recognized")
            if self._loaded is None or self.model is None or self._loaded != model_name:
                if self.model is not None:
                    del self.model
                    self._models[self._loaded]["unload_model"]()
                # if self.model is not None:
                #     try:
                #         self.model.to("cpu")
                #     except:
                #         log.warning(f"Could not move model to cpu")
                #     del self.model
                #     try:
                #         torch.cuda.empty_cache()
                #         gc.collect()
                #     except:
                #         log.warning(f"Could not empty GPU cache")
                log.info(f"Loading model {model_name}")
                self.model = self._models[model_name]["load_model"]()
                self._loaded = model_name
                log.info(f"Model {model_name} loaded")

    def _check_load_model(self, model_name: str):
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not recognized")
        
        if self._loaded != model_name:
            self._load_model(model_name=model_name)
            
    def answer(self, model_type: str, prompts: list[str], max_tokens: int):
        """
        Answer a question using the specified model type, prompts, and maximum tokens.
        Parameters:
        - model_type (str): The type of the model to use.
        - prompts (list[str]): A list of prompts to use for generating the answer.
        - max_tokens (int): The maximum number of tokens to generate for the answer.
        Returns:
        - str: The generated answer.
        Raises:
        - ValueError: If the specified model type is not recognized.
        - Exception: If there is an error while generating the answer.
        """

        self._check_load_model(model_name=model_type)
        
        try:
            with self._mlmodel_lock:
                response = self._models[model_type]["answer"](self.model, prompts, max_tokens)
                log.info(f"Answered: {response}")
                return response
        except Exception as e:
            log.error(f"Error answering: {e}")
            raise e

    def article_summary(self, model_type: str, articles: list[str], max_tokens: int, prompt: str):
        """
        Generate a summary for a list of articles using the specified model type.
        Args:
            model_type (str): The type of the model to use.
            articles (list[str]): A list of articles to summarize.
            max_tokens (int): The maximum number of tokens in the generated summary.
        Returns:
            list[str]: A list of summaries for the given articles.
        Raises:
            ValueError: If the specified model type is not recognized.
        Exception: If there is an error while generating the summaries.
        """
        self._check_load_model(model_name=model_type)
        try:
            with self._mlmodel_lock:
                return self._models[model_type]["article_summary"](
                    self.model, 
                    articles=articles, 
                    max_tokens=max_tokens, 
                    prompt=prompt
                    )
        except Exception as e:
            log.error(f"Error generating article summary: {e}")
            raise e
    
    def topic_name(self, model_type: str, keywords: list[str], prompt: str):
        self._check_load_model(model_name=model_type)
        try:
            with self._mlmodel_lock:
                return self._models[model_type]["topic_name"](
                    self.model, 
                    keywords=keywords, 
                    prompt=prompt
                    )
        except Exception as e:
            log.error(f"Error generating topic name: {e}")
            raise e
    
    def bullet_point_summary(self, model_type: str, articles: list[str], max_tokens: int, prompt: str):
        self._check_load_model(model_name=model_type)
        try:
            with self._mlmodel_lock:
                return self._models[model_type]["bullet_point_summary"](
                    self.model, 
                    articles=articles, 
                    max_tokens=max_tokens,
                    prompt=prompt,
                    )
        except Exception as e:
            log.error(f"Error generating bullet point summary: {e}")
            raise e

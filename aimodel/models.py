import logging
from pathlib import Path
from threading import Lock
from typing import Protocol
from gpt4all import GPT4All
from huggingface_hub import snapshot_download, login
from transformers import AutoModelForCausalLM, AutoTokenizer
from enum import Enum
import os
import torch

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


class LoadMLModel(Protocol):
    """
    Protocol for loading a machine learning model.
    The `LoadMLModel` protocol defines the interface for a callable object that loads a machine learning model.
    The `__call__` method should be implemented to return the loaded model.
    Returns:
        object: The loaded machine learning model.
    """

    def __call__(self) -> object: ...


class GPT4AllModel:
    """
    A class representing a GPT-4 model for generating responses to prompts.
    Attributes:
        lock (Lock): A lock object for thread safety.
        model (object): The GPT-4 model instance.
    """

    def __init__(self) -> None:
        self.lock = Lock()
        self.model = None

    def load_model(self) -> object:
        """
        Loads and returns the model if it exists, otherwise creates a new instance of GPT4All model and returns it.
        Returns:
            object: The loaded or newly created GPT4All model instance.
        """

        if self.model is not None:
            return self.model
        self.model = GPT4All(
            "Meta-Llama-3-8B-Instruct.Q4_0.gguf", n_threads=os.cpu_count()
        )
        return self.model

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
                    response.append(model.generate(prompt, max_tokens=max_tokens))
        return response


class MistralModel:
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
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        self._pre_download(mistral_path)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.to(self._device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    def _pre_download(self, mistral_folder: str | None = None):
        log.debug(f"Donwloding mistral model and storing in {mistral_folder}")
        if mistral_folder is not None:
            mistral_models_path = Path(mistral_folder).joinpath('mistral_models', '7B-v0.3')
        else:
            mistral_models_path = Path.home().joinpath('mistral_models', '7B-v0.3')
            
        mistral_models_path.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id="mistralai/Mistral-7B-v0.3", 
            allow_patterns=[
                "params.json", 
                "consolidated.safetensors", 
                "tokenizer.model.v3"], 
            local_dir=mistral_models_path
            )

    def load_model(self) -> object:
        return self.model

    def answer(self, model: object, prompts: list[str], max_tokens: int) -> list[str]:
        response = []
        with self.lock:
            for p_i, prompt in enumerate(prompts):
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs.to(self._device)
                outputs= model.generate(**inputs, max_new_tokens=max_tokens)
                model_response = self.tokenizer.decode(
                    outputs[0], 
                    skip_special_tokens=True)
                log.debug(f"Generated output: {model_response}")
                response.append(
                    model_response
                    )
        return response


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
        self._models = {}
        self._loaded = None
        self.model = None
        if load_default:
            self._gpt4all = GPT4AllModel()
            self.add_model(
                ModelType.GPT4ALL.value, 
                self._gpt4all.load_model, 
                self._gpt4all.answer
            )
            
            # TODO add mistral model
            if os.path.exists("/data"): 
                mistral_folder = "/data"
            else:
                mistral_folder = None
                
            self._mistral = MistralModel(mistral_path = mistral_path, hugging_face_api_key = hugging_face_api_key)
            self.add_model(
                ModelType.MISTRAL.value, 
                self._mistral.load_model, 
                self._mistral.answer)

    def add_model(self, model_name: str, load_model: LoadMLModel, answer: Answer):
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
            "answer": answer,
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

        log.debug(f"Loading model {model_name}")
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not recognized")
        if self._loaded is None or self.model is None:
            self.model = self._models[model_name]["load_model"]()
            self._loaded = model_name
            log.debug(f"Model {model_name} loaded")

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

        if model_type not in self._models:
            raise ValueError(f"Model {model_type} not recognized")

        if self._loaded != model_type:
            self._load_model(model_type)
        try:
            return self._models[model_type]["answer"](self.model, prompts, max_tokens)
        except Exception as e:
            log.error(f"Error answering: {e}")
            raise e

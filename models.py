import logging
from threading import Lock
from typing import Protocol
from gpt4all import GPT4All
from enum import Enum
import os

log = logging.getLogger('LLM')

class ModelType(Enum):
    GPT4ALL = 'gpt4all'
    MISTRAL = 'mistral'

class Answer(Protocol):
    def __call__(self, model: object, prompts:list[str], max_tokens:int) -> list[str]:
        ...

class LoadMLModel(Protocol):
    '''
    Load the model class to specify requirements.
    '''
    def __call__(self) -> object:
        ...

class GPT4AllModel:
    def __init__(self) -> None:
        self.lock = Lock()
        self.model = None
        
    def load_model(self) -> object:
        if self.model is not None:
            return self.model
        self.model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf", n_threads=os.cpu_count())
        return self.model
    
    def answer(self, model: object, prompts:list[str], max_tokens:int) -> list[str]:
        response = []
        with self.lock:
            with model.chat_session():
                for p_i, prompt in enumerate(prompts):
                    response.append(
                        model.generate(prompt, max_tokens=max_tokens)
                    )
        return response


class MistralModel:
    def __init(self) -> None:
        self.lock = Lock()
        self.model = None
    
    def load_model(self) -> object:
        if self.model is not None:
            return self.model
        raise NotImplementedError("Mistral model not implemented")
    
    def answer(self, model: object, prompts:list[str], max_tokens:int) -> list[str]:
        response = []
        with self.lock:
            with model.chat_session():
                for p_i, prompt in enumerate(prompts):
                    response.append(
                        model.generate(prompt, max_tokens=max_tokens)
                    )
        return response
    
    
class MLModel:
    def __init__(self, load_default: bool = True):
        log.info(f"Initializing MLModel")
        self._models = {}
        self._loaded = None
        self.model = None
        if load_default:
            self._gpt4all = GPT4AllModel()
            self.add_model(
                ModelType.GPT4ALL.value, 
                self._gpt4all.load_model, 
                self._gpt4all.answer)
            
            # TODO add mistral
            
            
    def add_model(
        self, 
        model_name: str,
        load_model: LoadMLModel,
        answer: Answer):
        if model_name in self._models:
            raise ValueError(f"Model {model_name} already exists")
        self._models[model_name] = {
            "name": model_name,
            "load_model": load_model,
            "answer": answer
        }
        
    @property
    def model_names(self):
        return list(self._models.keys())
    
    def _load_model(self, model_name: str = ModelType.GPT4ALL.value):
        log.debug(f"Loading model {model_name}")
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not recognized")
        if self._loaded is None or self.model is None:
            self.model = self._models[model_name]['load_model']()
            self._loaded = model_name
            log.debug(f"Model {model_name} loaded")
    
    def answer(self, model_type: str, prompts:list[str], max_tokens:int):
        if model_type not in self._models:
            raise ValueError(f"Model {model_type} not recognized")
        
        if self._loaded != model_type:
            self._load_model(model_type)
        try:
            return self._models[model_type]['answer'](self.model, prompts, max_tokens)
        except Exception as e:
            log.error(f"Error answering: {e}")
            raise e
        



    
# class Model:
#     def __init__(self, name: ModelType = ModelType.GPT4ALL):
#         log.info(f"Initializing model {name.value}")
#         self.name = name.value
#         self._loaded = False
#         self.lock_model = Lock()
#         self._load_model()
    
#     def add_model(
#         self, 
#         model_name: str,
#         load_model: LoadMLModel,
#         answer: Answer,):
#         if model_name in self._models:
#             raise ValueError(f"Model {model_name} already exists")
    
#     def _load_model(self):
#         log.debug("Loading model")
#         if self.name == ModelType.GPT4ALL.value:
#             model_name = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
#             log.debug(f"Loading model {model_name}")
#             self.model = GPT4All(model_name, n_threads=os.cpu_count()) # downloads / loads a 4.66GB LLM
#         elif self.name == ModelType.MISTRAL.value:
#             model_name = "Mistral"
#             raise NotImplementedError("Mistral model not implemented")
#         else:
#             raise ValueError(f"Model {self.name} not recognized")

#         self._loaded = True
        
#     def load_model(self):
#         if not self._loaded:
#             self._load_model()
    
#     def answer(self, model_type: str, prompts:list[str], max_tokens:int):
#         response = []
#         with self.lock_model:
#             if self.name != model_type:
#                 self.name = model_type
#                 self._loaded = False
#                 self.load_model()
#             if self.name == ModelType.GPT4ALL.value:
#                 with self.model.chat_session():
#                     for p_i, prompt in enumerate(prompts):
#                         response.append(
#                             # f"Test, index {p_i}",
#                             self.model.generate(prompt, max_tokens=max_tokens)
#                         )
#         return response


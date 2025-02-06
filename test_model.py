import logging
import unittest
from models import MLModel, ModelType

logging.basicConfig(level=logging.DEBUG)
prompt = '''You are a reporter. Your task is to create a summary of an article with a limit of 50 words. Provide only the summary, without any descriptions of the requested task.

# Article:
The former chief financial officer for Royal Bank of Canada is suing the bank for almost $50 million over claims of wrongful dismissal.

RBC announced on April 5 that it had fired Nadine Ahn after an internal review found evidence she was in an "undisclosed close personal relationship" with another employee who received preferential treatment, including promotion and compensation increases, which violated the bank's code of conduct.

A lawsuit filed by Ahn in the Ontario Superior Court of Justice on Thursday says there is no merit to the allegations, which it calls patently and categorically false.

The lawsuit states that Ahn denies providing preferential treatment to her colleague and that RBC's decision to fire her was tainted by gender-based stereotypes about friendships between women and men.

RBC spokesperson Gillian McArdle said in a statement that the facts are very clear that there was a significant breach of the bank's code of conduct, the claims in the lawsuit are without merit and the bank will vigorously defend against them in court.

The $48.9-million lawsuit, first reported by Bloomberg, includes seeking damages for wrongful dismissal, damages for defamation, punitive damages and other claims.
'''

class TestMLModel(unittest.TestCase):
    def setUp(self):
        # Set up any necessary objects or data before each test case
        self.model = MLModel(load_default=True)
        
    # def tearDown(self):
    #     # Clean up any resources after each test case
        
    def test_invalid_model(self):
        self.assertRaises(
            ValueError, 
            lambda: self.model.answer('unknown_model', [prompt], 1000))
        
    def test_answer(self):
        answer = self.model.answer(ModelType.GPT4ALL.value, [prompt], 1000)
        self.assertTrue(isinstance(answer, list))
        self.assertTrue(all([isinstance(ans, str) for ans in answer]))
    
    def test_all_models(self):
        model_names = self.model.model_names
        for model_name in model_names:
            logging.debug(f"Testing model {model_name}")
            answer = self.model.answer(model_name, [prompt], 1000)
            self.assertTrue(isinstance(answer, list))
            self.assertTrue(all([isinstance(ans, str) for ans in answer]))
        
if __name__ == '__main__':
    unittest.main()
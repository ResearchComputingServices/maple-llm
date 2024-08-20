from aimodel import MLModel, create_app

# Create your model with 2 functions: load_model and generate
class MyModel:
    def load_model(self):
        return self
    
    def generate(self):
        return "This is just a dummy answer."
    
# Create a function that will be used to answer the prompts
def answer(model: object, prompts:list[str], max_tokens:int) -> list[str]:
        response = []
        for prompt in prompts:
            response.append(model.generate())
        return response
    
# Create an instance of your model
my_model = MyModel()

# Create an instance of MLModel and add your model
models = MLModel(load_default=False)

# Add your model to the models
models.add_model(
    model_name="My own model",
    load_model=my_model.load_model,
    answer=answer,
    )

# Create the app
app = create_app(models=models)

if __name__ == '__main__':
    # Run the app
   app.run(port=5000, ssl_context='adhoc')

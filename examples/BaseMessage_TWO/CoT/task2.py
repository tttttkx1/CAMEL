from camel.models import ModelFactory
from camel.agents import TaskSpecifyAgent
from camel.types import ModelPlatformType, TaskType
from camel.prompts import TextPrompt
import os
from dotenv import load_dotenv
from openai import api_key

load_dotenv(dotenv_path="MyModel.env")
api_key = os.getenv('MODELSCOPE_SDK_TOKEN')

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="Qwen/Qwen2.5-72B-Instruct",
    url='https://api-inference.modelscope.cn/v1/',
    api_key=api_key
)

my_prompt_tamplate = TextPrompt(
    "Here is a task: I\'m a {occupation} and I want to {task}.Help me to make this task more specfic",
)

task_specify_agent = TaskSpecifyAgent(
    model=model,task_specify_prompt=my_prompt_tamplate,output_language="zh"
)

respose = task_specify_agent.run(
    task_prompt="get promotion",
    meta_dict=dict(occupation="Musician", task="get promotion")
)
print(respose)
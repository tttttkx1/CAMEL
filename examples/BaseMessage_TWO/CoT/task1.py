from asyncio import Task
from camel.models import ModelFactory
from camel.agents import TaskSpecifyAgent
from camel.types import ModelPlatformType, TaskType
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='MyModel.env')
api_key = os.getenv('MODELSCOPE_SDK_TOKEN')

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="Qwen/Qwen2.5-72B-Instruct",
    url='https://api-inference.modelscope.cn/v1/',
    api_key=api_key
)

task_specify_agent = TaskSpecifyAgent(
    model=model,task_type=TaskType.AI_SOCIETY,output_language="zh"
)
specify_task_prompt = task_specify_agent.run(
    task_prompt="Improving stage presence and performance skills",
    meta_dict=dict(
        assistant_role="You are a Musician",user_role="Student",word_limit=200
    ),
)
print(f"Specified task prompt:\n{specify_task_prompt}\n")

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.messages import BaseMessage

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

system_msg = "you are a helpful assistant that responds to user queries."

chat_agent = ChatAgent(model=model, system_message=system_msg,output_language="zh")

user_msg_with_meta = BaseMessage.make_user_message(
    role_name="User",
    content="Hello! Can you tell me something about Shandong University(weihai)?",
    meta_dict={
            "processing_time": 1.23, "api_version": "v2", "user_id": "1234567890",
            "user_preference": "simply", "language_setting": "zh-CN", "region": "Shandong"
    }
)
response = chat_agent.step(user_msg_with_meta)

print("Assistant Response:", response.msgs[0].content)
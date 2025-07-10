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

# 第一轮对话
user_msg1 = BaseMessage.make_user_message(
    role_name="User",
    content="What is the purpose of CAMEL?",
    meta_dict={
            "processing_time": 1.23, "api_version": "v2", "user_id": "1234567890",
            "user_preference": "simply", "language_setting": "zh-CN", "region": "Shandong",
            "turn": 1
    }
)
response1 = chat_agent.step(user_msg1)
print("Assistant Response 1:", response1.msgs[0].content)

# 第二轮对话
user_msg2 = BaseMessage.make_user_message(
    role_name="User",
    content="Can you provide an example of using CAMEL for multi-agent collaboration?",
    meta_dict={
            "processing_time": 1.56, "api_version": "v2", "user_id": "1234567890",
            "user_preference": "simply", "language_setting": "zh-CN", "region": "Shandong",
            "turn": 2
    }
)
response2 = chat_agent.step(user_msg2)
print("Assistant Response 2:", response2.msgs[0].content)
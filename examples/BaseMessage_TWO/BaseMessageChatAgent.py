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

# 创建系统消息，告诉ChatAgent自己的角色定位
system_msg = "You are a helpful assistant that responds to user queries."

# 实例化一个ChatAgent
chat_agent = ChatAgent(model=model, system_message=system_msg,output_language="en")

# 构造用户消息
user_msg = "Hello! Can you tell me something about CAMEL AI?"

# 将用户消息传给ChatAgent，并获取回复
response = chat_agent.step(user_msg)
print("Assistant Response:", response.msgs[0].content)

user_msg_with_meta = BaseMessage.make_user_message(
    role_name="User",
    content="Here is some extra context in the metadata.",
    meta_dict={
        "processing_time": 1.23, "api_version": "v2", "user_id": "1234567890"
    }
)
response = chat_agent.step(user_msg_with_meta)

print("Assistant Response:", response.msgs[0].content)

print("=== 消息元数据信息 ===")
print(f"用户ID: {user_msg_with_meta.meta_dict['user_id']}")
print(f"API版本: {user_msg_with_meta.meta_dict['api_version']}")
print(f"处理时间: {user_msg_with_meta.meta_dict['processing_time']}秒")

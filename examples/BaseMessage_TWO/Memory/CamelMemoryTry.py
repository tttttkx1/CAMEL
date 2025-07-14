from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='MyModel.env')
# 定义系统消息
sys_msg = "你是一个好奇的智能体，正在探索宇宙的奥秘。"

# 初始化agent 调用在线Qwen/Qwen2.5-72B-Instruct
api_key = os.getenv('MODELSCOPE_SDK_TOKEN')

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="Qwen/Qwen2.5-72B-Instruct",
    url='https://api-inference.modelscope.cn/v1/',
    api_key=api_key
)
agent = ChatAgent(system_message=sys_msg, model=model)

# 定义用户消息
usr_msg = "告诉我基于我们讨论的内容，哪个是第一个LLM多智能体框架？"

# 发送消息给agent
response = agent.step(usr_msg)

# 查看响应
print(response.msgs[0].content)

# ===== 迁移 CamelMemory.py 的 memory 初始化和写入内容 =====
from camel.memories import (
    LongtermAgentMemory,
    MemoryRecord,
    ScoreBasedContextCreator,
    ChatHistoryBlock,
    VectorDBBlock,
)
from camel.messages import BaseMessage
from camel.types import ModelType, OpenAIBackendRole
from camel.utils import OpenAITokenCounter
from camel.embeddings import SentenceTransformerEncoder

# 初始化 memory 系统
memory = LongtermAgentMemory(
    context_creator=ScoreBasedContextCreator(
        token_counter=OpenAITokenCounter(ModelType.GPT_4O_MINI),
        token_limit=2048,
    ),
    chat_history_block=ChatHistoryBlock(),
    vector_db_block=VectorDBBlock(
        embedding=SentenceTransformerEncoder(
            model_name="all-MiniLM-L6-v2",
        ),
    ),
)

# 创建 memory 记录
records = [
    MemoryRecord(
        message=BaseMessage.make_user_message(
            role_name="User",
            content="什么是CAMEL AI?"
        ),
        role_at_backend=OpenAIBackendRole.USER,
    ),
    MemoryRecord(
        message=BaseMessage.make_assistant_message(
            role_name="Agent",
            content="CAMEL-AI是第一个LLM多智能体框架,并且是一个致力于寻找智能体 scaling law 的开源社区。"
        ),
        role_at_backend=OpenAIBackendRole.ASSISTANT,
    ),
]

# 写入 memory
memory.write_records(records)
# ===== 迁移结束 =====

# 将memory赋值给agent
agent.memory = memory
# 发送消息给agent
response = agent.step(usr_msg)
# 查看响应
print(response.msgs[0].content)
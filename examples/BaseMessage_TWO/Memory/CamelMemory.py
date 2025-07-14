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

#initalize memory system
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

#2 create memory records
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

# 3 writting memory
memory.write_records(records)

context, token_cnt = memory.get_context()

print(context)
print(f'THE COMSUPTION OF TOKENS: {token_cnt}')
from camel.memories.blocks.vectordb_block import VectorDBBlock
from camel.memories.records import MemoryRecord
from camel.messages import BaseMessage
from camel.embeddings import SentenceTransformerEncoder
from camel.types import OpenAIBackendRole

#create a instance of ChatHistoryVectorBlock
#all-MiniLM-L6-v2 is a lightweight multilingual model
vector_db_block = VectorDBBlock(embedding=SentenceTransformerEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2"))

records = [
    MemoryRecord(message=BaseMessage.make_user_message(role_name="user", content="今天天气真好！"), role_at_backend=OpenAIBackendRole.USER),
    MemoryRecord(message=BaseMessage.make_user_message(role_name="user", content="你喜欢什么运动？"), role_at_backend=OpenAIBackendRole.USER),
    MemoryRecord(message=BaseMessage.make_user_message(role_name="user", content="今天天气不错，我们去散步吧。"), role_at_backend=OpenAIBackendRole.USER),
]

vector_db_block.write_records(records)

key_word = "天气"
retrieved_records = vector_db_block.retrieve(key_word, limit=3)

for record in retrieved_records:
    print(f"UUID: {record.memory_record.uuid}, Message: {record.memory_record.message.content}, Score: {record.score}")
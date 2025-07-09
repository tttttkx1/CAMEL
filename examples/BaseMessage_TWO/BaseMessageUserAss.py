from camel.messages import BaseMessage

# 创建用户消息
user_msg = BaseMessage.make_user_message(
    role_name="User_1",
    content="Hi, what can you do?"
)

# 创建助手消息
assistant_msg = BaseMessage.make_assistant_message(
    role_name="Assistant_1",
    content="I can help you with various tasks."
)

# 更新用户消息
updated_msg = user_msg.create_new_instance(
    content="Hi, can you tell me more about CAMEL?"
)

# 将助手消息转换为dict
assistant_msg_dict = assistant_msg.to_dict()

print("User Message:", user_msg)
print("Assistant Message:", assistant_msg)
print("Updated Message:", updated_msg)
print("Assistant Message Dict:", assistant_msg_dict)
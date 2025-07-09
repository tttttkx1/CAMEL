from camel.messages import BaseMessage
from camel.types import RoleType

# 创建一个简单的用户消息
message = BaseMessage(
    role_name="example_user",
    role_type=RoleType.USER,
    content="Hello, CAMEL!",
    meta_dict={} #添加必需的meta dict参数，即使为空也要提供，否则会报 TypeError
)

print(message)

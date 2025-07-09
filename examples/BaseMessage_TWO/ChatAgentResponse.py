from camel.responses import ChatAgentResponse
from camel.messages import BaseMessage
from camel.types import RoleType

# 创建一个 ChatAgentResponse 实例
response = ChatAgentResponse(
    msgs=[
        BaseMessage(
            role_name="Assistant",  # 助手的角色名称
            role_type=RoleType.ASSISTANT,  # 指定角色类型
            content="你好，我可以帮您做什么？",  # 消息内容
            meta_dict={}  # 提供一个空的元数据字典（可根据需要填充）
        )
    ],  
    terminated=False,  # 会话未终止
    info={"usage": {"prompt_tokens": 10, "completion_tokens": 15}}  # 附加信息
)

# 访问属性
messages = response.msgs  # 获取Agent生成的消息
is_terminated = response.terminated  # 会话是否终止
additional_info = response.info  # 获取附加信息

# 打印消息内容
print("消息内容:", messages[0].content)
# 打印会话是否终止
print("会话是否终止:", is_terminated)
# 打印附加信息
print("附加信息:", additional_info)
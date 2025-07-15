from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="MyModel.env")
api_key = os.getenv("MODELSCOPE_SDK_TOKEN")

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="Qwen/Qwen2.5-72B-Instruct",
    url='https://api-inference.modelscope.cn/v1/',
    api_key=api_key
)

sys_msg = "你是一个数学大师，擅长解决各种数学问题"

agent = ChatAgent(
    model=model,
    system_message=sys_msg,
    output_language="zh-CN",
    tools=[]
)

user_msg = "2的平方根是多少？"

response = agent.step(user_msg)
print(response.msg.content)

#--------------------------------------------------自定义数学工具----------------------------------------------------------------
from camel.toolkits import FunctionTool
import math

def calculate_square_root(x: float) -> float:
    r"""计算一个数的平方根。

    Args:
        x (float): 需要计算平方根的数字。

    Returns:
        float: 输入数字的平方根。
    """
    return math.sqrt(x)

square_root_tool = FunctionTool(calculate_square_root)

print(square_root_tool.get_function_name())
print(square_root_tool.get_function_description())

# 定义系统消息
sys_msg = "你是一个数学大师，擅长各种数学问题。当你遇到数学问题的时候，你要调用工具，将工具计算的结果作为答案"

tool_agent = ChatAgent(
    tools = [square_root_tool],
    system_message=sys_msg,
    model=model,
    output_language="中文")
    
# 重新发送消息给toolagent
response = tool_agent.step(user_msg)
print(response.msg.content)

print(response.info["tool_calls"])
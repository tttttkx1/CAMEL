from camel.societies import RolePlaying
from camel.utils import print_text_animated
from camel.models import ModelFactory
from camel.types import ModelPlatformType, TaskType, ModelType

import os
from dotenv import load_dotenv
from openai import api_key
from colorama import Fore

load_dotenv(dotenv_path="MyModel.env")
api_key = os.getenv('MODELSCOPE_SDK_TOKEN')

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="Qwen/Qwen2.5-72B-Instruct",
    url='https://api-inference.modelscope.cn/v1/',
    api_key=api_key
)

task_kwargs = {
    'task_prompt': '写一本关于AI社会的未来的书。',
    'with_task_specify': True,
    'task_specify_agent_kwargs': {'model': model}
}

user_role_kwargs = {
    'user_role_name': 'AI专家',
    'user_agent_kwargs': {'model': model}
}

assistant_role_kwargs = {
    'assistant_role_name': '对AI感兴趣的作家',
    'assistant_agent_kwargs': {'model': model}
}

society = RolePlaying(
    **task_kwargs,             # 任务参数
    **user_role_kwargs,        # 指令发送者的参数
    **assistant_role_kwargs,
    critic_role_name='human',#如果参数是human以外的参数，则会自动选择
    with_critic_in_the_loop=True,
    output_language='中文'   # 指令接收者的参数
)

def is_terminated(response):
    """
    当会话应该终止时给出对应信息。
    """
    if response.terminated:
        role = response.msg.role_type.name
        reason = response.info['termination_reasons']
        print(f'AI {role} 因为 {reason} 而终止')

    return response.terminated

def run(society, round_limit: int=10):

    # 获取AI助手到AI用户的初始消息
    input_msg = society.init_chat()

    # 开始互动会话
    for _ in range(round_limit):

        # 获取这一轮的两个响应
        assistant_response, user_response = society.step(input_msg)

        # 检查终止条件
        if is_terminated(assistant_response) or is_terminated(user_response):
            break

        # 获取结果
        print_text_animated(
            Fore.BLUE + f"AI用户的回答: {user_response.msg.content}\n"
        )
        # 检查任务是否结束
        if 'CAMEL_TASK_DONE' in user_response.msg.content:
            break
        print_text_animated(
            Fore.GREEN + f"AI助手的回答: {assistant_response.msg.content}\n"
        )

        # 获取下一轮的输入消息
        input_msg = assistant_response.msg

    return None

if __name__ == '__main__':
    run(society=society, round_limit=10)
"""
2.
代码生成任务：利用 CodePromptTemplateDict，创建一个任务Agent，帮助用户学习一门新的编程语言（例如 Python）。
要求 AI 逐步生成学习计划，包括基本概念、代码示例和练习题目。
"""

from camel.societies import role_playing
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.utils import print_text_animated

from dotenv import load_dotenv
import os

from colorama import Fore

load_dotenv(dotenv_path='MyModel.env')
# 获取API密钥并设置为OPENAI_API_KEY环境变量
api_key = os.getenv('MODELSCOPE_SDK_TOKEN')

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="Qwen/Qwen2.5-72B-Instruct",
    url='https://api-inference.modelscope.cn/v1/',
    api_key=api_key
)

def main(model=model, chat_turn_limit=50) -> None:
    # 任务的主要任务的提示词
    task_prompt = (
    "你是一名资深的 Python 教师，请为初学者分步骤讲解 Python 编程语言的基础知识。\n"
    "请按照以下方式组织教学：\n"
    "1. 先介绍基本语法概念（如变量、数据类型、控制流等）。\n"
    "2. 然后提供对应的代码示例。\n"
    "3. 最后给出练习题目供学生巩固所学内容。\n"
    "每一步都应以清晰的思维链方式进行推理，并输出详细的教学内容。"
    )

    # 为AI设置一个角色
    role_playing_session = role_playing.RolePlaying(
        assistant_role_name="Python 教师",
        assistant_agent_kwargs=dict(model=model),
        user_role_name="学生",
        user_agent_kwargs=dict(model=model),
        task_prompt=task_prompt,
        with_task_specify=True,
        task_specify_agent_kwargs=dict(model=model),
        output_language='中文'
    )

    #先将系统信息打印出来，包括assistant和user的系统信息
    print(
        Fore.GREEN 
        + f"AI助手的系统信息:\n{role_playing_session.assistant_sys_msg}\n"
    )
    print(
        Fore.BLUE
        + f"AI用户的系统信息:\n{role_playing_session.user_sys_msg}\n"
    )

    print(Fore.YELLOW + f"原始任务提示:\n{task_prompt}\n")
    print(
        Fore.CYAN
        + "指定的任务提示:"
        + f"\n{role_playing_session.specified_task_prompt}\n"
    )
    print(Fore.RED + f"最终任务提示:\n{role_playing_session.task_prompt}\n")

    #开始进入问答阶段
    n = 0
    input_msg = role_playing_session.init_chat()
    while n < chat_turn_limit:
        n += 1
        assistant_response, user_response = role_playing_session.step(input_msg)

        #当对话中止时，出现的信息会打印出来
        if assistant_response.terminated:
            print(
                Fore.GREEN
                + (
                    "AI助手终止。原因: "
                    f"{assistant_response.info['termination_reasons']}."
                )
            )
            break
        if user_response.terminated:
            print(
                Fore.BLUE
                + (
                    "AI用户终止。原因: "
                    f"{user_response.info['termination_reasons']}."
                )
            )

        #这里是正式的回答内容
        print_text_animated(
            Fore.BLUE + f"AI用户: {user_response.msg.content}\n"
        )
        print_text_animated(
            Fore.GREEN + f"AI助手: {assistant_response.msg.content}\n"
        )

        if "CAMEL_TASK_DONE" in user_response.msg.content:
            break
        
        input_msg = assistant_response.msg


if __name__ == "__main__":
    main(model=model, chat_turn_limit=50)
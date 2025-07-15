"""
1. 
角色扮演任务Agent：使用 AISocietyPromptTemplateDict，创建一个角色扮演任务Agent。
假设你想让 AI 扮演一个“健康顾问”，为一个“患者”提供饮食和锻炼建议。
请用思维链方式分解整个建议过程，逐步提供健康方案。
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

def main(model=model, chat_turn_limit=50)-> None:
    #任务的主要任务的提示词
    task_prompt = (
        "你是一名健康顾问，请为一位患者分步骤（思维链）提供科学的饮食和锻炼建议。"
        "请先收集患者基本信息（如年龄、性别、体重、健康目标），"
        "再分析其饮食和锻炼现状，结合健康目标，逐步推理并输出详细的饮食和锻炼方案。"
    )
    #为AI设置一个角色
    role_play_session = role_playing.RolePlaying(
        assistant_role_name="健康顾问",
        assistant_agent_kwargs=dict(model=model),
        user_role_name="患者",
        user_agent_kwargs=dict(model=model),
        task_prompt=task_prompt,
        with_task_specify=True,
        task_specify_agent_kwargs=dict(model=model),
        output_language='中文'
    )

    #先将系统信息打印出来，包括assistant和user的系统信息
    print(
        Fore.GREEN 
        + f"AI助手的系统信息:\n{role_play_session.assistant_sys_msg}\n"
    )
    print(
        Fore.BLUE
        + f"AI用户的系统信息:\n{role_play_session.user_sys_msg}\n"
    )

    print(Fore.YELLOW + f"原始任务提示:\n{task_prompt}\n")
    print(
        Fore.CYAN
        + "指定的任务提示:"
        + f"\n{role_play_session.specified_task_prompt}\n"
    )
    print(Fore.RED + f"最终任务提示:\n{role_play_session.task_prompt}\n")

    #开始进入问答阶段
    n = 0
    input_msg = role_play_session.init_chat()
    while n < chat_turn_limit:
        n += 1
        assistant_response, user_response = role_play_session.step(input_msg)

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
            Fore.BLUE
            + f"AI用户: {user_response.msg.content}\n"
        )
        print_text_animated(
            Fore.GREEN
            + f"AI助手: {assistant_response.msg.content}\n"
        )

        if "CAMEL_TASK_DONE" in user_response.msg.content:
            break
        
        input_msg = assistant_response.msg

if __name__ == "__main__":
    main(model=model, chat_turn_limit=50)

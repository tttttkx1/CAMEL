"""
创建一个自己的工具，结合CAMEL内置的其他工具，使用RolePlaying并让Agent帮你完成一个任务。
"""

from camel.societies import role_playing
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.utils import print_text_animated
from camel.toolkits import FunctionTool, SearchToolkit, MathToolkit

from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
from colorama import Fore

def calculate_moving_average_signal(prices: list)-> str:
    """
    根据收盘价列表计算简单移动平均线（SMA）
    当短期均线 > 长期均线时，返回 "买入"
    否则返回 "卖出"
    """
    df = pd.DataFrame(prices, columns=['Close'])
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    latest_signal = "买入" if df['SMA_5'].iloc[-1] > df['SMA_20'].iloc[-1] else "卖出"
    return latest_signal

#将自定义的函数添加到工具列表中
stock_trading_tool = FunctionTool(calculate_moving_average_signal)

#将内置的工具和自定义工具添加到工具列表中
tools_list = [
    *SearchToolkit().get_tools(),
    *MathToolkit().get_tools(),
    stock_trading_tool
]

#------------------------------------以上是自定义工具的部分------------------------------------
# 获取API密钥并设置为OPENAI_API_KEY环境变量
load_dotenv(dotenv_path='MyModel.env')
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY', '')# 谷歌API密钥,调用Google搜索工具时需要
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
        "假设现在是2024年，"
        "请使用Google搜索工具查询特朗普集团的成立年份，并计算出其当前年龄。"
        "然后再将这个年龄加上10年。"
        "最后，使用股票交易工具判断是否应该买入或卖出特朗普集团相关的股票。"
    )

    # 为AI设置一个角色
    role_play_session = role_playing.RolePlaying(
        assistant_role_name="金融分析师",
        assistant_agent_kwargs=dict(model=model, tools=tools_list),
        user_role_name="投资者",
        user_agent_kwargs=dict(model=model),
        task_prompt=task_prompt,
        with_task_specify=False,
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
        print_text_animated(
            Fore.GREEN + f"AI助手的回答: {assistant_response.msg.content}\n",
            #speed=0.05
        )
        print_text_animated(
            Fore.BLUE + f"AI用户的回答: {user_response.msg.content}\n",
            #speed=0.05
        )
        
        if "CAMEL_TASK_DONE" in assistant_response.info:
            print(
                Fore.GREEN
                + "AI助手已完成任务。"
            )
            break
        input_msg = assistant_response.msg

if __name__ == "__main__":
    main(model=model, chat_turn_limit=50)
# FinalWork1_1.py（已经经过修改并和第三章的StockBot联动）

from camel.societies import role_playing
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.utils import print_text_animated

from dotenv import load_dotenv
import os

from colorama import Fore

# 加载 .env 文件中的环境变量
load_dotenv(dotenv_path='MyModel.env')

# 获取 API 密钥并设置为环境变量
api_key = os.getenv('MODELSCOPE_SDK_TOKEN')

# 创建模型实例
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="Qwen/Qwen2.5-72B-Instruct",
    url='https://api-inference.modelscope.cn/v1/',
    api_key=api_key
)

import sys
import os

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".."))
# 添加项目根目录到Python路径
sys.path.append(project_root)

# 尝试最可能的导入方式
try:
    from examples.CamelAndPractice_THREE.StockTradeRobot import StockTradingBot
    print("成功使用标准导入")
except ImportError:
    stock_trade_path = os.path.join(project_root, "examples", "CamelAndPractice_THREE", "StockTradeRobot.py")
    if os.path.exists(stock_trade_path):
        sys.path.append(os.path.dirname(stock_trade_path))
        from StockTradeRobot import StockTradingBot
        print("成功使用相对路径导入")
    else:
        print(f"错误：StockTradeRobot.py 文件不存在于预期位置: {stock_trade_path}")
        raise FileNotFoundError(f"无法找到StockTradeRobot.py文件 at {stock_trade_path}")

def main(model=model, chat_turn_limit=50) -> None:
    # 任务的主要任务的提示词
    task_prompt = (
        "你是一名Python程序员，请为一位股票交易员开发一个基于机器学习的股票交易机器人。"
        "该机器人需要能够自动分析市场趋势、执行买卖操作，并实时调整策略以优化投资组合。"
        "请先确定所需的数据源和技术栈，"
        "再设计机器学习模型架构和交易策略逻辑，"
        "最后实现完整的数据处理、模型训练和交易执行流程。"
    )
    
    # 创建StockTradingBot实例
    bot = StockTradingBot(ticker='AAPL', period='2y')
    
    # 获取历史数据并训练模型
    bot.fetch_data()
    accuracy = bot.train_model()
    print(f"模型准确率: {accuracy:.2f}")
    
    # 为AI设置一个角色
    role_play_session = role_playing.RolePlaying(
        assistant_role_name="Python Programmer",
        assistant_agent_kwargs=dict(model=model),
        user_role_name="Stock Trader",
        user_agent_kwargs=dict(model=model),
        task_prompt=task_prompt,
        with_task_specify=True,
        task_specify_agent_kwargs=dict(model=model),
        output_language='中文'
    )

    # 先将系统信息打印出来，包括assistant和user的系统信息
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

    # 开始进入问答阶段
    n = 0
    input_msg = role_play_session.init_chat()
    while n < chat_turn_limit:
        n += 1
        assistant_response, user_response = role_play_session.step(input_msg)

        # 当对话中止时，出现的信息会打印出来
        if assistant_response.terminated:
            print(
                Fore.GREEN
                + (
                    "AI助手终止。原因: "
                    f"{assistant_response.info['termination_reasons']}.)"
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
            break

        # 这里是正式的回答内容
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

    # 使用训练好的模型生成最终建议
    latest_data = bot.data.iloc[-1][['MA5', 'MA20', 'RSI']]
    final_signal = bot.predict_signal(latest_data)
    print(f"\n最终交易建议: {final_signal}")
    print(f"模型准确率: {accuracy:.2f}")

if __name__ == "__main__":
    main(model=model, chat_turn_limit=50)

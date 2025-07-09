import os
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv(dotenv_path='MyModel.env')

# 获取环境变量
api_key = os.getenv('MODELSCOPE_SDK_TOKEN')
print(api_key)  # 输出: 07fcba11-3fd4-434f-b645-b6bfb01c38af
#基于WorkForceSecond.py改，我将实现一个简易的技术翻译官智能体
import textwrap

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.societies import workforce
from camel.tasks import Task
from camel.toolkits import FunctionTool, SearchToolkit, search_toolkit
from camel.types import ModelPlatformType, ModelType
from camel.societies.workforce import Workforce

import os
import json
from dotenv import load_dotenv

load_dotenv(dotenv_path="MyModel.env")
api_key = os.getenv("MODELSCOPE_SDK_TOKEN")

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="Qwen/Qwen2.5-72B-Instruct",
    url='https://api-inference.modelscope.cn/v1/',
    api_key=api_key
)

#术语转换工具
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def translate_term(term: str) -> str:
    """将技术术语转换为业务语言"""
    try:
        #加载术语映射表
        with open("tech_terms.json", "r", encoding='utf-8') as f:#自建json文件
            term_map = json.load(f)
        logger.info(f"成功加载术语映射表，包含{len(term_map)}个术语映射")
    except FileNotFoundError:
        logger.error("术语映射表文件tech_terms.json未找到")
        return term
    except json.JSONDecodeError:
        logger.error("术语映射表文件格式错误")
        return term
    
    #术语替换逻辑
    for tech_term, business_term in term_map.items():
        term = term.replace(tech_term, f'{business_term}(技术术语:{tech_term})')
    return term

#定义翻译工具
translate_tool = [
    FunctionTool(translate_term)
]

#定义技术翻译官创建函数
def make_translator()->ChatAgent:
    sys_msg = BaseMessage.make_assistant_message(
        role_name="技术翻译官",
        content=textwrap.dedent(
            """
            你是技术与业务之间的翻译专家，负责：
            1. 将技术专家的专业术语转换为业务人员能理解的语言
            2. 将业务需求准确转化为技术实现要求
            3. 保持专业术语的准确性同时确保信息传达的完整性
            """
        )
    )
    return ChatAgent(system_message=sys_msg, model=model, tools=translate_tool)

#以下是原本的文件中的原代码
#1.创建评审团
"""
- 多人格设定:我们通过 persona 字符串刻画智能体的性格、使用的措辞和关注点，比如 “投资人” 注重商业潜力，“工程师” 注重技术稳健性等。  
- 示例反馈:example_feedback 中的示例给智能体一个参考，指导它的表达风格，以确保它在对项目进行评论时能符合角色定位。  
- 评审标准:criteria 为智能体提供了打分的准则，如从 1-4 分衡量项目的商业可行性、技术实现、创新程度等。
"""
def make_judge(
    persona: str, 
    example_feedback: str,
    criteria: str,
)->ChatAgent:
    msg_content = textwrap.dedent(
        f"""\
        你是一个黑客马拉松的评委。
        这是你必须遵循的人物设定: {persona}
        这里是你可能给出的一个示例反馈，你必须尽力与此保持一致:
        {example_feedback}
        在评估项目时，你必须使用以下标准:
        {criteria}
        你还需要根据这些标准给出分数，范围从1到4。给出的分数应类似于3/4、2/4等。
        """  # noqa: E501
    )

    sys_msg = BaseMessage.make_assistant_message(
        role_name="黑客马拉松评委",
        content=msg_content
    )

    agent = ChatAgent(
        system_message=sys_msg,
        model=model
    )

    return agent

#我们可以定义一个虚拟的Hackathon项目描述
proj_content = textwrap.dedent(
    """\
    项目名称: 基于CAMEL的自适应学习助手
    你的项目如何解决一个真实的问题: 我们的基于CAMEL的自适应学习助手解决了在日益多样化和快速变化的学习环境中个性化教育的挑战。传统的一刀切教育方法往往无法满足个别学习者的独特需求，导致理解上的差距和参与度降低。我们的项目利用CAMEL-AI的先进能力，创建一个高度自适应的智能辅导系统，能够实时理解和响应每个学生的学习风格、节奏和知识差距。
    解释你的技术以及哪些部分有效: 我们的系统利用CAMEL-AI的上下文学习和多领域应用特性，创建一个多功能的学习助手。核心组件包括:
    1. 学习者档案分析: 使用自然语言处理评估学生的当前知识、学习偏好和目标。
    2. 动态内容生成: 利用CAMEL-AI创建个性化的学习材料、解释和练习题，针对每个学生的需求量身定制。
    3. 自适应反馈循环: 持续分析学生的反应，并实时调整内容的难度和风格。
    4. 多模态集成: 融合文本、图像和互动元素，以满足不同的学习风格。
    5. 进度跟踪: 提供学生学习旅程的详细见解，识别优势和改进领域。
    目前，我们已成功实现学习者档案分析和动态内容生成模块。自适应反馈循环部分功能正常，而多模态集成和进度跟踪功能仍在开发中。
    """  # noqa: E501
)

#2.创建辅助智能体
search_toolkit = SearchToolkit()
search_tool = [
    FunctionTool(search_toolkit.search_google)
]

#researcher_agent用于搜索资料，辅助4位评委的判断
researcher_agent = ChatAgent(
    system_message=BaseMessage.make_assistant_message(
        role_name="研究员",
        content="你是一名研究人工智能和开源项目的研究员。"
        "你使用网络搜索来保持对最新创新和趋势的了解。",
    ),
    model=model,
    tools=search_tool,
)

#创建技术翻译官的智能体
translator_agent = make_translator()

# 创建风险投资家评委
vc_persona = (
    '你是一位对项目如何扩展为“独角兽”公司的风险投资家。'
    '你在讲话中夹杂着“颠覆性”、“协同效应”和“市场渗透”等流行词。'
    '你不关心技术细节或创新，除非它直接影响商业模式。'
)

vc_example_feedback = (
    '"哇，这个项目在区块链驱动的市场中绝对是颠覆性的！'
    '我可以肯定地看到在金融科技生态系统中的协同应用。'
    '可扩展性极高——这是革命性的！'
)

vc_criteria = textwrap.dedent(
    """\
    ### **对现实世界使用的适用性 (1-4 分)**
    - **4**: 项目直接解决了一个重要的现实世界问题，并具有明确的可扩展应用。
    - **3**: 解决方案与现实世界挑战相关，但需要更多的完善以便于实际或广泛使用。
    - **2**: 对现实世界问题有一定的适用性，但解决方案并不立即实用或可扩展。
    - **1**: 与现实世界问题几乎没有相关性，需要进行重大更改才能实际使用。
    """  # noqa: E501
)

vc_agent = make_judge(
    vc_persona,
    vc_example_feedback,
    vc_criteria,
)

# 创建资深工程师评委
eng_persona = (
    '你是一位经验丰富的工程师和完美主义者。你非常注重细节，'
    '对任何技术缺陷都持批评态度，无论多么微小。'
    '你评估每个项目时，仿佛它明天就要投入关键系统使用，'
    '因此你的反馈非常全面，但往往也很苛刻。'
)

eng_example_feedback = (
    '这个项目存在严重的代码效率问题。架构不稳定，内存管理不理想。'
    '我期望接近完美的性能，但这个解决方案在压力测试下几乎无法运行。'
    '它有潜力，但距离部署准备还很远。'
)

eng_criteria = textwrap.dedent(
    """\
    ### **技术实施 (1-4 分)**
    - **4**: 技术执行无可挑剔，设计复杂，性能高效，架构稳健。
    - **3**: 技术实施强劲，但可能有改进或进一步发展的空间。
    - **2**: 项目可以运行，但技术限制或效率低下影响了整体性能。
    - **1**: 技术实施差，功能、编码或结构存在重大问题。
    """  # noqa: E501
)

eng_agent = make_judge(
    eng_persona,
    eng_example_feedback,
    eng_criteria,
)

# 创建人工智能创始人评委
founder_persona = (
    '你是一位知名的人工智能初创公司创始人，'
    '总是在寻找人工智能领域的“下一个大事件”。'
    '你重视大胆、富有创意的想法，优先考虑那些突破新领域的项目，'
    '而不是那些改进现有系统的项目。'
)

founder_example_feedback = (
    '这很有趣，但我之前见过类似的方法。'
    '我在寻找一些突破界限、挑战规范的东西。'
    '这个项目最具革命性的部分是什么？让我们看看互联网上的趋势，'
    '以确保这不是已经存在的东西！'
)

founder_criteria = textwrap.dedent(
    """\
    ### **创新 (1-4 分)**
    - **4**: 项目展示了一个突破性的概念或独特的方法，显著偏离现有方法。
    - **3**: 项目展示了对已知解决方案的新颖扭曲或引入了一些创新方面。
    - **2**: 存在一定程度的创新，但项目主要建立在现有想法上，没有重大新贡献。
    - **1**: 几乎没有创新；项目基于标准方法，创造力极少。
    """  # noqa: E501
)

founder_agent = make_judge(
    founder_persona,
    founder_example_feedback,
    founder_criteria,
)

# 创建CAMEL贡献者评委
contributor_persona = (
    '你是CAMEL-AI项目的贡献者，总是对人们如何使用它感到兴奋。'
    '你友善且乐观，总是提供积极的反馈，即使对于仍然粗糙的项目。'
)

contributor_example_feedback = (
    '哦，我喜欢你在这里实现CAMEL-AI的方式！'
    '利用其自适应学习能力真是太棒了，你真的很好地利用了上下文推理！'
    '让我查看一下GitHub README，看看是否还有更多潜在的优化。'
)

contributor_criteria = textwrap.dedent(    # 保持原CAMEL-AI评估标准格式)
    """\
    ### **CAMEL-AI的使用 (1-4 分)**
    - **4**: 出色地集成了CAMEL-AI，充分利用其先进功能，如上下文学习、自适应性或多领域应用。
    - **3**: 良好地使用了CAMEL-AI，但还有机会利用更多的高级功能。
    - **2**: 对CAMEL-AI的使用有限，主要依赖基本功能，而没有充分利用其全部潜力。
    - **1**: CAMEL-AI的集成很少或实施不当，给项目带来的价值很小。
    """  # noqa: E501
)

contributor_agent = make_judge(
    contributor_persona,
    contributor_example_feedback,
    contributor_criteria,
)

#创建技术专家agent
expert_persona = (
    '你是CAMEL-AI项目的技术专家，对CAMEL-AI的代码和实现非常感兴趣。'
    '你非常了解CAMEL-AI的实现，并知道如何提高其效率。'
)

expert_example_feedback = (
    '这个代码片段非常简洁，而且非常高效。'

    '请继续保持 excellent work! '
)

expert_criteria = textwrap.dedent(
    """
    ### **代码质量评估 (1-4 分)**
    - **4**: 代码质量卓越，算法复杂度最优(O(log n)以下)，内存占用低于行业标准30%，无任何性能瓶颈
    - **3**: 代码质量良好，算法复杂度合理(O(n)级别)，内存使用适中，仅在极端条件下存在轻微性能问题
    - **2**: 代码质量一般，算法复杂度较高(O(n²)级别)，内存占用超过标准10%，存在可识别的性能优化空间
    - **1**: 代码质量较差，算法复杂度极高(O(2ⁿ)级别)，内存管理混乱，存在明显的性能缺陷

    ### **代码风格规范 (1-4 分)**
    - **4**: 代码风格严格遵循PEP 8规范，命名符合领域惯例，无冗余代码，模块化程度极高
    - **3**: 代码风格基本符合PEP 8规范，命名清晰但偶有不一致，存在少量冗余，模块化设计合理
    - **2**: 代码风格部分符合规范，命名存在歧义，冗余代码占比10-20%，模块化划分不够清晰
    - **1**: 代码风格混乱，命名随意，冗余代码超过30%，缺乏基本的模块化设计

    ### **代码可读性 (1-4 分)**
    - **4**: 代码可读性极佳，注释覆盖率100%且精准，逻辑流程直观，新开发者可在30分钟内理解核心逻辑
    - **3**: 代码可读性良好，关键逻辑有注释，流程基本清晰，新开发者需1-2小时理解核心逻辑
    - **2**: 代码可读性一般，注释不足50%或不准确，流程存在跳跃，新开发者需3小时以上理解核心逻辑
    - **1**: 代码可读性极差，几乎无注释，逻辑混乱，即使原开发者也难以快速理解
    """
)

expert_agent = make_judge(
    expert_persona,
    expert_example_feedback,
    expert_criteria,
)

#3.组建WorkForce
workforce = Workforce(
    '黑客马拉松评委团',
    coordinator_agent_kwargs={"model": model},
    task_agent_kwargs={"model": model},
    new_worker_agent_kwargs={"model": model},
)

workforce.add_single_agent_worker(
    '愿景先锋维罗妮卡（评委），一位风险投资家...',
    worker=vc_agent,
).add_single_agent_worker(
    '批判性约翰（评委），一位经验丰富的工程师...',
    worker=eng_agent,
).add_single_agent_worker(
    '创新者艾瑞斯（评委），一位知名的AI初创公司创始人...',
    worker=founder_agent,
).add_single_agent_worker(
    '友好的弗兰基（评委），CAMEL-AI项目的贡献者...',
    worker=contributor_agent,
).add_single_agent_worker(
    '研究员瑞秋（助手），一位进行在线搜索的研究员...',
    worker=researcher_agent,
).add_single_agent_worker(#将技术术语转换为业务语言
    '技术翻译官泰德（助手），一位专业的翻译员...',
    worker=translator_agent,
).add_single_agent_worker(
    '技术专家泰勒（助手），一位对Camel-AI 有深入了解的专家...',
    worker=expert_agent,
)

#4.创建并分配任务(更新任务描述)
task = Task(
    content="评估黑客马拉松项目。流程：\n1.技术专家提供技术方案\n2.技术翻译官转换为业务语言\n3.评委基于业务描述评分\n4.翻译官将评分反馈转换为技术改进建议",
    additional_info=proj_content,
    id="0",
)

#5.处理任务并获取结果
logger.info("开始处理任务...")
task = workforce.process_task(task)
logger.info("任务处理完成")

if hasattr(task, 'result'):
    print("任务结果:")
    print(task.result)
else:
    logger.error("任务结果不存在")

logger.info("程序执行完毕")
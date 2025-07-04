# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import argparse
import logging
import os
import sys

from camel.typing import ModelType

root = os.path.dirname(__file__)
sys.path.append(root)

from chatdev.chat_chain import ChatChain

try:
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
    from openai.types.chat.chat_completion_message import FunctionCall

    openai_new_api = True  # new openai api version
except ImportError:
    openai_new_api = False  # old openai api version
    print(
        "Warning: Your OpenAI version is outdated. \n "
        "Please update as specified in requirement.txt. \n "
        "The old API interface is deprecated and will no longer be supported.")


def get_config(company):
    """
    return configuration json files for ChatChain
    user can customize only parts of configuration json files, other files will be left for default
    Args:
        company: customized configuration name under CompanyConfig/

    Returns:
        path to three configuration jsons: [config_path, config_phase_path, config_role_path]
    """
    config_dir = os.path.join(root, "CompanyConfig", company)
    default_config_dir = os.path.join(root, "CompanyConfig", "Default")

    config_files = [
        "ChatChainConfig.json",
        "PhaseConfig.json",
        "RoleConfig.json"
    ]

    config_paths = []

    for config_file in config_files:
        company_config_path = os.path.join(config_dir, config_file)
        default_config_path = os.path.join(default_config_dir, config_file)

        if os.path.exists(company_config_path):
            config_paths.append(company_config_path)
        else:
            config_paths.append(default_config_path)

    return tuple(config_paths)


parser = argparse.ArgumentParser(description='argparse')
parser.add_argument('--config', type=str, default="Default",
                    help="Name of config, which is used to load configuration under CompanyConfig/")
parser.add_argument('--org', type=str, default="DefaultOrganization",
                    help="Name of organization, your software will be generated in WareHouse/name_org_timestamp")
parser.add_argument('--task', type=str, default="Develop a basic Gomoku game.",
                    help="Prompt of software")
parser.add_argument('--name', type=str, default="Gomoku",
                    help="Name of software, your software will be generated in WareHouse/name_org_timestamp")
parser.add_argument('--model', type=str, default="GPT_3_5_TURBO",
                    help="GPT Model, choose from {'GPT_3_5_TURBO', 'GPT_4', 'GPT_4_TURBO', 'GPT_4O', 'GPT_4O_MINI'}")
parser.add_argument('--path', type=str, default="",
                    help="Your file directory, ChatDev will build upon your software in the Incremental mode")
parser.add_argument('--judge_model', type=str, default="GPT_4O_MINI",
                    help="Model used for judgment of generated code")
parser.add_argument('--api_source', type=str,
                    help="API source, choose from {'hk', 'hub', 'dashscope'}")
args = parser.parse_args()

if args.api_source == "hk":
    os.environ['BASE_URL'] = os.environ.get('OPENAI_HK_BASE_URL')
    os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_HK_API_KEY')
elif args.api_source == "hub":
    os.environ['BASE_URL'] = os.environ.get('OPENAI_HUB_BASE_URL')
    os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_HUB_API_KEY')
elif args.api_source == "dashscope":
    os.environ['BASE_URL'] = os.environ.get('DASHSCOPE_BASE_URL')
    os.environ['OPENAI_API_KEY'] = os.environ.get('DASHSCOPE_API_KEY')
else:
    raise ValueError("Invalid api_source. Choose from {'hk', 'hub', 'dashscope'}")

# Start ChatDev

# ----------------------------------------
#          Init ChatChain
# ----------------------------------------
config_path, config_phase_path, config_role_path = get_config(args.config)
args2type = {'GPT_3_5_TURBO': ModelType.GPT_3_5_TURBO,
             'GPT_4': ModelType.GPT_4,
            #  'GPT_4_32K': ModelType.GPT_4_32k,
             'GPT_4_TURBO': ModelType.GPT_4_TURBO,
            #  'GPT_4_TURBO_V': ModelType.GPT_4_TURBO_V
            'GPT_4O': ModelType.GPT_4O,
            'GPT_4O_MINI': ModelType.GPT_4O_MINI,
            'GPT_4_1_NANO': ModelType.GPT_4_1_NANO,
            'GPT_4_1_MINI': ModelType.GPT_4_1_MINI,
            'GPT_4_1': ModelType.GPT_4_1,
            'GROK_3': ModelType.GROK_3,
            'QWEN_MAX': ModelType.QWEN_MAX,
            'CLAUDE_SONNET_4': ModelType.CLAUDE_SONNET_4, # claude-sonnet-4-20250514
            'DEEPSEEK_V3': ModelType.DEEPSEEK_V3, # deepseek-v3-0324
            'GEMINI_2_5_PRO': ModelType.GEMINI_2_5_PRO,
            'GEMINI_2_5_FLASH': ModelType.GEMINI_2_5_FLASH, # gemini-2.5-flash-preview-04-17
            'GEMINI_2_5_FLASH_ALL': ModelType.GEMINI_2_5_FLASH_ALL, # gemini-2.5-flash-all
             }
if openai_new_api:
    args2type['GPT_3_5_TURBO'] = ModelType.GPT_3_5_TURBO_NEW

chat_chain = ChatChain(config_path=config_path,
                       config_phase_path=config_phase_path,
                       config_role_path=config_role_path,
                       task_prompt=args.task,
                       project_name=args.name,
                       org_name=args.org,
                       model_type=args2type[args.model],
                       judge_model_type=args2type[args.judge_model],
                       code_path=args.path)

# ----------------------------------------
#          Init Log
# ----------------------------------------
logging.basicConfig(filename=chat_chain.log_filepath, level=logging.INFO,
                    format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%Y-%d-%m %H:%M:%S', encoding="utf-8")

# ----------------------------------------
#          Pre Processing
# ----------------------------------------

chat_chain.pre_processing()

# ----------------------------------------
#          Personnel Recruitment
# ----------------------------------------

chat_chain.make_recruitment()

# ----------------------------------------
#          Chat Chain
# ----------------------------------------

chat_chain.execute_chain()

# ----------------------------------------
#          Post Processing
# ----------------------------------------

chat_chain.post_processing()

import os
import re
from abc import ABC, abstractmethod

import json
import ast
import concurrent.futures
import shutil
from tenacity import retry, stop_after_attempt, wait_random_exponential

from camel.agents import RolePlaying
from camel.messages import ChatMessage
from camel.typing import TaskType, ModelType
from chatdev.chat_env import ChatEnv
from chatdev.statistics import get_info
from chatdev.utils import log_visualize, log_arguments

from camel.schema import JudgeTriplet, JudgeTripletList
from chatdev.utils import CodeParser


class Phase(ABC):

    def __init__(self,
                 assistant_role_name,
                 user_role_name,
                 phase_prompt,
                 role_prompts,
                 phase_name,
                 model_type,
                 judge_model_type,
                 log_filepath):
        """

        Args:
            assistant_role_name: who receives chat in a phase
            user_role_name: who starts the chat in a phase
            phase_prompt: prompt of this phase
            role_prompts: prompts of all roles
            phase_name: name of this phase
        """
        self.seminar_conclusion = None
        self.assistant_role_name = assistant_role_name
        self.user_role_name = user_role_name
        self.phase_prompt = phase_prompt
        self.phase_env = dict()
        self.phase_name = phase_name
        self.assistant_role_prompt = role_prompts[assistant_role_name]
        self.user_role_prompt = role_prompts[user_role_name]
        self.ceo_prompt = role_prompts["Chief Executive Officer"]
        self.counselor_prompt = role_prompts["Counselor"]
        self.max_retries = 3
        self.reflection_prompt = """Here is a conversation between two roles: {conversations} {question}"""
        self._model_type = model_type
        self._judge_model_type = judge_model_type
        self.log_filepath = log_filepath
        self.complete_output_filter = ["DetailedTaskReviewComment", "CodeLocalization"]
        self.judge_phase_filter = [
            "TaskDecomposition",
            "CodeLocalization",
            "DetailedTaskReviewComment",
            "UnitTestCoding",
            "UnitTestExecution",
            "ModificationPlanGeneration",
            "UnitTestCodeModification",
            "UnifiedCodeModification",
            "CodeClean",
            "TestErrorSummary",
            "TestModification",]

    @property
    def model_type(self):
        if self.phase_name in self.judge_phase_filter:
            return self._judge_model_type
        return self._model_type

    @log_arguments
    def chatting(
            self,
            chat_env,
            task_prompt: str,
            assistant_role_name: str,
            user_role_name: str,
            phase_prompt: str,
            phase_name: str,
            assistant_role_prompt: str,
            user_role_prompt: str,
            task_type=TaskType.CHATDEV,
            need_reflect=False,
            with_task_specify=False,
            model_type=ModelType.GPT_3_5_TURBO,
            memory=None,
            placeholders=None,
            chat_turn_limit=10
    ) -> str:
        """

        Args:
            chat_env: global chatchain environment
            task_prompt: user query prompt for building the software
            assistant_role_name: who receives the chat
            user_role_name: who starts the chat
            phase_prompt: prompt of the phase
            phase_name: name of the phase
            assistant_role_prompt: prompt of assistant role
            user_role_prompt: prompt of user role
            task_type: task type
            need_reflect: flag for checking reflection
            with_task_specify: with task specify
            model_type: model type
            placeholders: placeholders for phase environment to generate phase prompt
            chat_turn_limit: turn limits in each chat

        Returns:

        """

        if placeholders is None:
            placeholders = {}
        assert 1 <= chat_turn_limit <= 100

        if not chat_env.exist_employee(assistant_role_name):
            raise ValueError(f"{assistant_role_name} not recruited in ChatEnv.")
        if not chat_env.exist_employee(user_role_name):
            raise ValueError(f"{user_role_name} not recruited in ChatEnv.")

        # init role play
        role_play_session = RolePlaying(
            assistant_role_name=assistant_role_name,
            user_role_name=user_role_name,
            assistant_role_prompt=assistant_role_prompt,
            user_role_prompt=user_role_prompt,
            task_prompt=task_prompt,
            task_type=task_type,
            with_task_specify=with_task_specify,
            memory=memory,
            model_type=model_type,
            background_prompt=chat_env.config.background_prompt
        )

        # log_visualize("System", role_play_session.assistant_sys_msg)
        # log_visualize("System", role_play_session.user_sys_msg)

        # start the chat
        _, input_user_msg = role_play_session.init_chat(None, placeholders, phase_prompt)
        seminar_conclusion = None

        # handle chats
        # the purpose of the chatting in one phase is to get a seminar conclusion
        # there are two types of conclusion
        # 1. with "<INFO>" mark
        # 1.1 get seminar conclusion flag (ChatAgent.info) from assistant or user role, which means there exist special "<INFO>" mark in the conversation
        # 1.2 add "<INFO>" to the reflected content of the chat (which may be terminated chat without "<INFO>" mark)
        # 2. without "<INFO>" mark, which means the chat is terminated or normally ended without generating a marked conclusion, and there is no need to reflect
        for i in range(chat_turn_limit):
            # start the chat, we represent the user and send msg to assistant
            # 1. so the input_user_msg should be assistant_role_prompt + phase_prompt
            # 2. then input_user_msg send to LLM and get assistant_response
            # 3. now we represent the assistant and send msg to user, so the input_assistant_msg is user_role_prompt + assistant_response
            # 4. then input_assistant_msg send to LLM and get user_response
            # all above are done in role_play_session.step, which contains two interactions with LLM
            # the first interaction is logged in role_play_session.init_chat
            assistant_response, user_response = role_play_session.step(input_user_msg, chat_turn_limit == 1)

            conversation_meta = "**" + assistant_role_name + "<->" + user_role_name + " on : " + str(
                phase_name) + ", turn " + str(i) + "**\n\n"

            # TODO: max_tokens_exceeded errors here
            if isinstance(assistant_response.msg, ChatMessage):
                # we log the second interaction here
                log_visualize(role_play_session.assistant_agent.role_name,
                              conversation_meta + "[" + role_play_session.user_agent.system_message.content + "]\n\n" + assistant_response.msg.content)
                if role_play_session.assistant_agent.info:
                    seminar_conclusion = assistant_response.msg.content
                    break
                if assistant_response.terminated:
                    break

            if isinstance(user_response.msg, ChatMessage):
                # here is the result of the second interaction, which may be used to start the next chat turn
                log_visualize(role_play_session.user_agent.role_name,
                              conversation_meta + "[" + role_play_session.assistant_agent.system_message.content + "]\n\n" + user_response.msg.content)
                if role_play_session.user_agent.info:
                    seminar_conclusion = user_response.msg.content
                    break
                if user_response.terminated:
                    break

            # continue the chat
            if chat_turn_limit > 1 and isinstance(user_response.msg, ChatMessage):
                input_user_msg = user_response.msg
            else:
                break

        # conduct self reflection
        if need_reflect:
            if seminar_conclusion in [None, ""]:
                seminar_conclusion = "<INFO> " + self.self_reflection(task_prompt, role_play_session, phase_name,
                                                                      chat_env)
            if "recruiting" in phase_name:
                if "Yes".lower() not in seminar_conclusion.lower() and "No".lower() not in seminar_conclusion.lower():
                    seminar_conclusion = "<INFO> " + self.self_reflection(task_prompt, role_play_session,
                                                                          phase_name,
                                                                          chat_env)
            elif seminar_conclusion in [None, ""]:
                seminar_conclusion = "<INFO> " + self.self_reflection(task_prompt, role_play_session, phase_name,
                                                                      chat_env)
        else:
            seminar_conclusion = assistant_response.msg.content

        log_visualize("**[Seminar Conclusion]** for {}:\n\n {}".format(phase_name, seminar_conclusion))
        # 返回完整的LLM输出结果，目前主要是针对Review
        if phase_name in self.complete_output_filter:
            return seminar_conclusion
        seminar_conclusion = seminar_conclusion.split("<INFO>")[-1].strip()
        return seminar_conclusion

    def self_reflection(self,
                        task_prompt: str,
                        role_play_session: RolePlaying,
                        phase_name: str,
                        chat_env: ChatEnv) -> str:
        """

        Args:
            task_prompt: user query prompt for building the software
            role_play_session: role play session from the chat phase which needs reflection
            phase_name: name of the chat phase which needs reflection
            chat_env: global chatchain environment

        Returns:
            reflected_content: str, reflected results

        """
        messages = role_play_session.assistant_agent.stored_messages if len(
            role_play_session.assistant_agent.stored_messages) >= len(
            role_play_session.user_agent.stored_messages) else role_play_session.user_agent.stored_messages
        messages = ["{}: {}".format(message.role_name, message.content.replace("\n\n", "\n")) for message in messages]
        messages = "\n\n".join(messages)

        if "recruiting" in phase_name:
            question = """Answer their final discussed conclusion (Yes or No) in the discussion without any other words, e.g., "Yes" """
        elif phase_name == "DemandAnalysis":
            question = """Answer their final product modality in the discussion without any other words, e.g., "PowerPoint" """
        elif phase_name == "LanguageChoose":
            question = """Conclude the programming language being discussed for software development, in the format: "*" where '*' represents a programming language." """
        elif phase_name == "EnvironmentDoc":
            question = """According to the codes and file format listed above, write a requirements.txt file to specify the dependencies or packages required for the project to run properly." """
        else:
            raise ValueError(f"Reflection of phase {phase_name}: Not Assigned.")

        # Reflections actually is a special phase between CEO and counselor
        # They read the whole chatting history of this phase and give refined conclusion of this phase
        reflected_content = \
            self.chatting(chat_env=chat_env,
                          task_prompt=task_prompt,
                          assistant_role_name="Chief Executive Officer",
                          user_role_name="Counselor",
                          phase_prompt=self.reflection_prompt,
                          phase_name="Reflection",
                          assistant_role_prompt=self.ceo_prompt,
                          user_role_prompt=self.counselor_prompt,
                          placeholders={"conversations": messages, "question": question},
                          need_reflect=False,
                          memory=chat_env.memory,
                          chat_turn_limit=1,
                          model_type=self.model_type)

        if "recruiting" in phase_name:
            if "Yes".lower() in reflected_content.lower():
                return "Yes"
            return "No"
        else:
            return reflected_content

    @abstractmethod
    def update_phase_env(self, chat_env):
        """
        update self.phase_env (if needed) using chat_env, then the chatting will use self.phase_env to follow the context and fill placeholders in phase prompt
        must be implemented in customized phase
        the usual format is just like:
        ```
            self.phase_env.update({key:chat_env[key]})
        ```
        Args:
            chat_env: global chat chain environment

        Returns: None

        """
        pass

    @abstractmethod
    def update_chat_env(self, chat_env) -> ChatEnv:
        """
        update chan_env based on the results of self.execute, which is self.seminar_conclusion
        must be implemented in customized phase
        the usual format is just like:
        ```
            chat_env.xxx = some_func_for_postprocess(self.seminar_conclusion)
        ```
        Args:
            chat_env:global chat chain environment

        Returns:
            chat_env: updated global chat chain environment

        """
        pass

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6)
    )
    def execute(self, chat_env, chat_turn_limit, need_reflect) -> ChatEnv:
        """
        execute the chatting in this phase
        1. receive information from environment: update the phase environment from global environment
        2. execute the chatting
        3. change the environment: update the global environment using the conclusion
        Args:
            chat_env: global chat chain environment
            chat_turn_limit: turn limit in each chat
            need_reflect: flag for reflection

        Returns:
            chat_env: updated global chat chain environment using the conclusion from this phase execution

        """
        self.update_phase_env(chat_env)
        self.seminar_conclusion = \
            self.chatting(chat_env=chat_env,
                          task_prompt=chat_env.env_dict['task_prompt'],
                          need_reflect=need_reflect,
                          assistant_role_name=self.assistant_role_name,
                          user_role_name=self.user_role_name,
                          phase_prompt=self.phase_prompt,
                          phase_name=self.phase_name,
                          assistant_role_prompt=self.assistant_role_prompt,
                          user_role_prompt=self.user_role_prompt,
                          chat_turn_limit=chat_turn_limit,
                          placeholders=self.phase_env,
                          memory=chat_env.memory,
                          model_type=self.model_type)
        # 如果返回了<INFO> Finished，则重新生成一次，避免审查随机性同时控制调用次数
        # 如果想要进一步节省token用量，注释掉这个if语句
        # if self.phase_name in self.complete_output_filter:
        #     if "<INFO> Finished" in self.seminar_conclusion and self.seminar_conclusion.split("<INFO> Finished")[0].strip() == "":
        #         self.seminar_conclusion = \
        #             self.chatting(chat_env=chat_env,
        #                           task_prompt=chat_env.env_dict['task_prompt'],
        #                           need_reflect=need_reflect,
        #                           assistant_role_name=self.assistant_role_name,
        #                           user_role_name=self.user_role_name,
        #                           phase_prompt=self.phase_prompt,
        #                           phase_name=self.phase_name,
        #                           assistant_role_prompt=self.assistant_role_prompt,
        #                           user_role_prompt=self.user_role_prompt,
        #                           chat_turn_limit=chat_turn_limit,
        #                           placeholders=self.phase_env,
        #                           memory=chat_env.memory,
        #                           model_type=self.model_type)
        chat_env = self.update_chat_env(chat_env)
        return chat_env

    @staticmethod
    def _extract_filename_from_line(lines):
        file_name = ""
        for candidate in re.finditer(r"(\w+\.\w+)", lines, re.DOTALL):
            file_name = candidate.group()
            file_name = file_name.lower()
        return file_name

    @staticmethod
    def _extract_filename_from_code(code):
        file_name = ""
        regex_extract = r"class (\S+?):\n"
        matches_extract = re.finditer(regex_extract, code, re.DOTALL)
        for match_extract in matches_extract:
            file_name = match_extract.group(1)
        file_name = file_name.lower().split("(")[0] + ".py"
        return file_name

    @staticmethod
    def _parse_list(content):
        info_content = content.strip()
        try:
            if info_content.startswith('[') and info_content.endswith(']'):
                return json.loads(info_content)
        except (json.JSONDecodeError, ValueError):
            pass
        list_pattern = r'\[([^\[\]]*(?:\[[^\]]*\][^\[\]]*)*)\]'
        match = re.search(list_pattern, info_content, re.DOTALL)
        if match:
            list_str = match.group(0)
            try:
                return json.loads(list_str)
            except (json.JSONDecodeError, ValueError):
                pass
        try:
            return ast.literal_eval(info_content)
        except (ValueError, SyntaxError):
            pass
        return []


class DemandAnalysis(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        pass

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0:
            chat_env.env_dict['modality'] = self.seminar_conclusion.split("<INFO>")[-1].lower().replace(".", "").strip()
        return chat_env


class LanguageChoose(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "description": chat_env.env_dict['task_description'],
                               "modality": chat_env.env_dict['modality'],
                               "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        # if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
        #     chat_env.env_dict['language'] = self.seminar_conclusion.split("<INFO>")[-1].lower().replace(".", "").strip()
        # elif len(self.seminar_conclusion) > 0:
        #     chat_env.env_dict['language'] = self.seminar_conclusion
        # else:
        #     chat_env.env_dict['language'] = "Python"
        chat_env.env_dict['language'] = chat_env.config.language
        return chat_env

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6)
    )
    def execute(self, chat_env, chat_turn_limit, need_reflect) -> ChatEnv:
        self.update_phase_env(chat_env)
        chat_env = self.update_chat_env(chat_env)
        return chat_env


class Coding(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        gui = "" if not chat_env.config.gui_design \
            else "The software should be equipped with graphical user interface (GUI) so that user can visually and graphically use it; so you must choose a GUI framework (e.g., in Python, you can implement GUI via tkinter, Pygame, Flexx, PyGUI, etc,)."
        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "description": chat_env.env_dict['task_description'],
                               "modality": chat_env.env_dict['modality'],
                               "ideas": chat_env.env_dict['ideas'],
                               "language": chat_env.env_dict['language'],
                               "gui": gui})

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.update_codes(self.seminar_conclusion)
        if len(chat_env.codes.codebooks.keys()) == 0:
            raise ValueError("No Valid Codes.")
        chat_env.rewrite_codes("Finish Coding")
        log_visualize(
            "**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'], self.log_filepath)))
        return chat_env


class CodeComplete(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "modality": chat_env.env_dict['modality'],
                               "ideas": chat_env.env_dict['ideas'],
                               "language": chat_env.env_dict['language'],
                               "codes": chat_env.get_codes(),
                               "unimplemented_file": ""})
        unimplemented_file = ""
        for filename in self.phase_env['pyfiles']:
            code_content = open(os.path.join(chat_env.env_dict['directory'], filename)).read()
            lines = [line.strip() for line in code_content.split("\n") if line.strip() == "pass"]
            if len(lines) > 0 and self.phase_env['num_tried'][filename] < self.phase_env['max_num_implement']:
                unimplemented_file = filename
                break
        self.phase_env['num_tried'][unimplemented_file] += 1
        self.phase_env['unimplemented_file'] = unimplemented_file

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.update_codes(self.seminar_conclusion)
        if len(chat_env.codes.codebooks.keys()) == 0:
            raise ValueError("No Valid Codes.")
        chat_env.rewrite_codes("Code Complete #" + str(self.phase_env["cycle_index"]) + " Finished")
        log_visualize(
            "**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'], self.log_filepath)))
        return chat_env


class TestErrorSummary(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        # chat_env.generate_images_from_codes()
        (exist_bugs_flag, test_reports) = chat_env.exist_bugs()
        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "modality": chat_env.env_dict['modality'],
                               "ideas": chat_env.env_dict['ideas'],
                               "language": chat_env.env_dict['language'],
                               "codes": chat_env.get_codes(),
                               "test_reports": test_reports,
                               "exist_bugs_flag": exist_bugs_flag})
        log_visualize("**[Test Reports]**:\n\n{}".format(test_reports))

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['error_summary'] = self.seminar_conclusion
        chat_env.env_dict['test_reports'] = self.phase_env['test_reports']

        return chat_env

    def execute(self, chat_env, chat_turn_limit, need_reflect) -> ChatEnv:
        self.update_phase_env(chat_env)
        if "ModuleNotFoundError" in self.phase_env['test_reports']:
            chat_env.fix_module_not_found_error(self.phase_env['test_reports'])
            log_visualize(
                f"Software Test Engineer found ModuleNotFoundError:\n{self.phase_env['test_reports']}\n")
            pip_install_content = ""
            for match in re.finditer(r"No module named '(\S+)'", self.phase_env['test_reports'], re.DOTALL):
                module = match.group(1)
                pip_install_content += "{}\n```{}\n{}\n```\n".format("cmd", "bash", f"pip install {module}")
                log_visualize(f"Programmer resolve ModuleNotFoundError by:\n{pip_install_content}\n")
            self.seminar_conclusion = "nothing need to do"
        else:
            self.seminar_conclusion = \
                self.chatting(chat_env=chat_env,
                              task_prompt=chat_env.env_dict['task_prompt'],
                              need_reflect=need_reflect,
                              assistant_role_name=self.assistant_role_name,
                              user_role_name=self.user_role_name,
                              phase_prompt=self.phase_prompt,
                              phase_name=self.phase_name,
                              assistant_role_prompt=self.assistant_role_prompt,
                              user_role_prompt=self.user_role_prompt,
                              memory=chat_env.memory,
                              chat_turn_limit=chat_turn_limit,
                              placeholders=self.phase_env,
                              model_type=self.model_type)
        chat_env = self.update_chat_env(chat_env)
        return chat_env


class TestModification(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "modality": chat_env.env_dict['modality'],
                               "ideas": chat_env.env_dict['ideas'],
                               "language": chat_env.env_dict['language'],
                               "test_reports": chat_env.env_dict['test_reports'],
                               "error_summary": chat_env.env_dict['error_summary'],
                               "codes": chat_env.get_codes()
                               })

    def update_chat_env(self, chat_env) -> ChatEnv:
        if "```".lower() in self.seminar_conclusion.lower():
            chat_env.update_codes(self.seminar_conclusion)
            chat_env.rewrite_codes("Test #" + str(self.phase_env["cycle_index"]) + " Finished")
            log_visualize(
                "**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'], self.log_filepath)))
        return chat_env


class UnitTestCoding(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "modality": chat_env.env_dict['modality'],
                               "language": chat_env.env_dict['language'],
                               "codes": chat_env.get_codes(),
                               "judge_triplet": chat_env.env_dict['judge_triplet'], })

    def update_chat_env(self, chat_env, judge_triplet) -> ChatEnv:
        try:
            judge_triplet.unit_test_code = CodeParser.parse_code(block="", text=self.seminar_conclusion)
        except ValueError as e:
            log_visualize(f"Error parsing unit test code: {e}")
        return chat_env

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6)
    )
    def _process_single_triplet(self, chat_env, judge_triplet, need_reflect, chat_turn_limit):
        placeholders = {
            "task": self.phase_env['task'],
            "modality": self.phase_env['modality'],
            "language": self.phase_env['language'],
            "codes": self.phase_env['codes'],
            "task_item": judge_triplet.subtask,
            "code_snippets": judge_triplet.code_snippets
        }
        seminar_conclusion = self.chatting(
            chat_env=chat_env,
            task_prompt=chat_env.env_dict['task_prompt'],
            need_reflect=need_reflect,
            assistant_role_name=self.assistant_role_name,
            user_role_name=self.user_role_name,
            phase_prompt=self.phase_prompt,
            phase_name=self.phase_name,
            assistant_role_prompt=self.assistant_role_prompt,
            user_role_prompt=self.user_role_prompt,
            chat_turn_limit=chat_turn_limit,
            placeholders=placeholders,
            memory=chat_env.memory,
            model_type=self.model_type
        )
        return seminar_conclusion, judge_triplet

    def execute(self, chat_env, chat_turn_limit, need_reflect) -> ChatEnv:
        self.update_phase_env(chat_env)
        if self.phase_env["cycle_index"] != 1:
            return chat_env
        active_triplets = [triplet for triplet in self.phase_env["judge_triplet"].triplets
                           if not triplet.is_complete and triplet.need_unit_test]
        if not active_triplets:
            return chat_env
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_triplets)) as executor:
            future_to_triplet = {
                executor.submit(
                    self._process_single_triplet, chat_env, triplet, need_reflect, chat_turn_limit
                ): triplet for triplet in active_triplets
            }
            for future in concurrent.futures.as_completed(future_to_triplet):
                result = future.result()
                self.seminar_conclusion, judge_triplet = result
                chat_env = self.update_chat_env(chat_env, judge_triplet)
        return chat_env


class UnitTestExecution(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "modality": chat_env.env_dict['modality'],
                               "language": chat_env.env_dict['language'],
                               "codes": chat_env.get_codes(),
                               "judge_triplet": chat_env.env_dict['judge_triplet'], })

    def update_chat_env(self, chat_env, judge_triplet) -> ChatEnv:
        judge_triplet.unit_test_analysis = self.seminar_conclusion
        return chat_env

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6)
    )
    def _process_single_triplet(self, chat_env, judge_triplet, index, need_reflect, chat_turn_limit):
        test_filename = f"test_for_subtask_{index}.py"
        test_filepath = os.path.join(chat_env.env_dict["directory"], test_filename)

        with open(test_filepath, "w", encoding="utf-8") as f:
            f.write(judge_triplet.unit_test_code)
        log_visualize(f"Unit test code written to: {test_filepath}")

        (exist_bugs_flag, unit_test_report) = chat_env.run_test_code(test_filename)
        judge_triplet.unit_test_result = not exist_bugs_flag
        judge_triplet.unit_test_report = unit_test_report

        if not exist_bugs_flag:
            log_visualize(f"**[Test Info]**\n\nAI User (Software Test Engineer):\nUnit Test Pass!\n")
            return None

        placeholders = {
            "task": chat_env.env_dict['task_prompt'],
            "modality": chat_env.env_dict['modality'],
            "language": chat_env.env_dict['language'],
            "codes": chat_env.get_codes(),
            "task_item": judge_triplet.subtask,
            "code_snippets": judge_triplet.code_snippets,
            "unit_test_code": judge_triplet.unit_test_code,
            "unit_test_report": unit_test_report,
        }

        seminar_conclusion = self.chatting(
            chat_env=chat_env,
            task_prompt=chat_env.env_dict['task_prompt'],
            need_reflect=need_reflect,
            assistant_role_name=self.assistant_role_name,
            user_role_name=self.user_role_name,
            phase_prompt=self.phase_prompt,
            phase_name=self.phase_name,
            assistant_role_prompt=self.assistant_role_prompt,
            user_role_prompt=self.user_role_prompt,
            chat_turn_limit=chat_turn_limit,
            placeholders=placeholders,
            memory=chat_env.memory,
            model_type=self.model_type
        )

        conclusion = CodeParser.parse_blocks(seminar_conclusion)
        if conclusion.get("Problem Source") is None or conclusion.get("Analysis") is None or conclusion.get("Code Modification") is None:
            raise ValueError("Invalid seminar conclusion format.")

        return seminar_conclusion, judge_triplet

    def execute(self, chat_env, chat_turn_limit, need_reflect) -> ChatEnv:
        self.update_phase_env(chat_env)
        active_triplets = [
            (index, triplet) for index, triplet in enumerate(self.phase_env["judge_triplet"].triplets)
            if not triplet.is_complete and triplet.need_unit_test and triplet.unit_test_code
        ]
        if not active_triplets:
            return chat_env
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_triplets)) as executor:
            future_to_triplet = {
                executor.submit(
                    self._process_single_triplet,
                    chat_env,
                    triplet,
                    index,
                    need_reflect,
                    chat_turn_limit
                ): (index, triplet) for index, triplet in active_triplets
            }
            for future in concurrent.futures.as_completed(future_to_triplet):
                result = future.result()
                if result is not None:
                    self.seminar_conclusion, judge_triplet = result
                    chat_env = self.update_chat_env(chat_env, judge_triplet)
        return chat_env


class TaskDecomposition(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        if not isinstance(chat_env.env_dict.get('judge_triplet'), JudgeTripletList):
            chat_env.env_dict['judge_triplet'] = JudgeTripletList()
        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "judge_triplet": chat_env.env_dict['judge_triplet']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        task_list = self._parse_list(self.seminar_conclusion)
        if not task_list:
            raise ValueError("No valid task list found in the seminar conclusion.")
        for subtask in task_list:
            chat_env.env_dict['judge_triplet'].triplets.append(JudgeTriplet(subtask=subtask))
        return chat_env


class CodeLocalization(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "modality": chat_env.env_dict['modality'],
                               "language": chat_env.env_dict['language'],
                               "codes": chat_env.get_codes(),
                               "judge_triplet": chat_env.env_dict['judge_triplet']})

    def update_chat_env(self, chat_env, judge_triplet) -> ChatEnv:
        if "```".lower() in self.seminar_conclusion.lower():
            code_blocks = []
            regex = r"(.+?)\n```.*?\n(.*?)```"
            matches = re.finditer(regex, self.seminar_conclusion, re.DOTALL)
            for match in matches:
                code = match.group(2)
                group1 = match.group(1)
                filename = self._extract_filename_from_line(group1)
                if filename == "":
                    filename = self._extract_filename_from_code(code)
                if not filename:
                    continue
                formatted_block = f"{filename}\n```\n{code}\n```"
                code_blocks.append(formatted_block)
            judge_triplet.code_snippets = "\n\n".join(code_blocks)
        elif "<INFO> Not found".lower() in self.seminar_conclusion.lower():
            judge_triplet.code_snippets = "Not found"
        else:
            judge_triplet.code_snippets = self.seminar_conclusion.strip()
        return chat_env

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6)
    )
    def _process_single_triplet(self, chat_env, judge_triplet, need_reflect, chat_turn_limit):
        placeholders = {
            "task": self.phase_env['task'],
            "modality": self.phase_env['modality'],
            "language": self.phase_env['language'],
            "codes": self.phase_env["codes"],
            "task_item": judge_triplet.subtask
        }
        seminar_conclusion = self.chatting(
            chat_env=chat_env,
            task_prompt=chat_env.env_dict['task_prompt'],
            need_reflect=need_reflect,
            assistant_role_name=self.assistant_role_name,
            user_role_name=self.user_role_name,
            phase_prompt=self.phase_prompt,
            phase_name=self.phase_name,
            assistant_role_prompt=self.assistant_role_prompt,
            user_role_prompt=self.user_role_prompt,
            chat_turn_limit=chat_turn_limit,
            placeholders=placeholders,
            memory=chat_env.memory,
            model_type=self.model_type
        )
        return seminar_conclusion, judge_triplet

    def execute(self, chat_env, chat_turn_limit, need_reflect) -> ChatEnv:
        self.update_phase_env(chat_env)
        active_triplets = [triplet for triplet in self.phase_env["judge_triplet"].triplets
                           if not triplet.is_complete]
        if not active_triplets:
            return chat_env
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_triplets)) as executor:
            future_to_triplet = {
                executor.submit(
                    self._process_single_triplet, chat_env, triplet, need_reflect, chat_turn_limit
                ): triplet for triplet in active_triplets
            }
            for future in concurrent.futures.as_completed(future_to_triplet):
                result = future.result()
                self.seminar_conclusion, judge_triplet = result
                chat_env = self.update_chat_env(chat_env, judge_triplet)
        return chat_env


class DetailedTaskReviewComment(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "modality": chat_env.env_dict['modality'],
                               "language": chat_env.env_dict['language'],
                               "codes": chat_env.get_codes(),
                               "judge_triplet": chat_env.env_dict['judge_triplet'], })

    def update_chat_env(self, chat_env, judge_triplet, conclusion) -> ChatEnv:
        judge_triplet.review_comments = conclusion["Code Review Comment"]
        judge_triplet.review_result = "pass" in conclusion["Code Review Result"].lower()
        judge_triplet.need_unit_test = "true" in conclusion["Unit Test Necessity"].lower()
        return chat_env

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6)
    )
    def _process_single_triplet(self, chat_env, judge_triplet, need_reflect, chat_turn_limit):
        placeholders = {
            "task": self.phase_env['task'],
            "modality": self.phase_env['modality'],
            "language": self.phase_env['language'],
            "codes": self.phase_env["codes"],
            "task_item": judge_triplet.subtask,
            "code_snippets": judge_triplet.code_snippets
        }
        seminar_conclusion = self.chatting(
            chat_env=chat_env,
            task_prompt=chat_env.env_dict['task_prompt'],
            need_reflect=need_reflect,
            assistant_role_name=self.assistant_role_name,
            user_role_name=self.user_role_name,
            phase_prompt=self.phase_prompt,
            phase_name=self.phase_name,
            assistant_role_prompt=self.assistant_role_prompt,
            user_role_prompt=self.user_role_prompt,
            chat_turn_limit=chat_turn_limit,
            placeholders=placeholders,
            memory=chat_env.memory,
            model_type=self.model_type
        )
        conclusion = CodeParser.parse_blocks(seminar_conclusion)
        if conclusion.get("Code Review Comment") is None or conclusion.get("Code Review Result") is None or conclusion.get("Unit Test Necessity") is None:
            raise ValueError("Invalid seminar conclusion format.")
        return conclusion, judge_triplet

    def execute(self, chat_env, chat_turn_limit, need_reflect) -> ChatEnv:
        self.update_phase_env(chat_env)
        active_triplets = [triplet for triplet in self.phase_env["judge_triplet"].triplets
                           if not triplet.is_complete]
        if not active_triplets:
            return chat_env
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_triplets)) as executor:
            future_to_triplet = {
                executor.submit(
                    self._process_single_triplet, chat_env, triplet, need_reflect, chat_turn_limit
                ): triplet for triplet in active_triplets
            }
            for future in concurrent.futures.as_completed(future_to_triplet):
                result = future.result()
                conclusion, judge_triplet = result
                chat_env = self.update_chat_env(chat_env, judge_triplet, conclusion)
        return chat_env


class ModificationPlanGeneration(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        combined_evidence = ""
        for judge_triplet in chat_env.env_dict['judge_triplet'].triplets:
            if judge_triplet.is_complete:
                continue
            triplet_evidence = ""
            if not judge_triplet.review_result:
                triplet_evidence += f">>> [Reference] Code Review Comment:\n\n{judge_triplet.review_comments}\n\n"
            if not judge_triplet.need_unit_test or not judge_triplet.unit_test_code:
                continue
            if not judge_triplet.unit_test_result:
                conclusion = CodeParser.parse_blocks(judge_triplet.unit_test_analysis)
                if "Source Code".lower() in conclusion["Problem Source"].lower():
                    triplet_evidence += f">>> [Reference] Unit Test Report:\n\n{judge_triplet.unit_test_report}\n\n"
                    analysis = f"## Analysis\n{conclusion['Analysis']}\n## Code Modification\n{conclusion['Code Modification']}\n\n"
                    triplet_evidence += f">>> [Reference] Unit Test Analysis:\n\n{analysis}\n\n"
            if triplet_evidence.strip():
                if combined_evidence:
                    combined_evidence += f"--------------------------------------------------\n\n"
                combined_evidence += f">>> [Key Evidence] Subtask:\n\n{judge_triplet.subtask}\n\n"
                combined_evidence += triplet_evidence

        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "modality": chat_env.env_dict['modality'],
                               "language": chat_env.env_dict['language'],
                               "codes": chat_env.get_codes(),
                               "evidence": combined_evidence})

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['modification_plan'] = self.seminar_conclusion
        return chat_env

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6)
    )
    def execute(self, chat_env, chat_turn_limit, need_reflect) -> ChatEnv:
        self.update_phase_env(chat_env)
        if not self.phase_env.get("evidence", "").strip():
            chat_env.env_dict['modification_plan'] = ""
            return chat_env
        self.seminar_conclusion = \
            self.chatting(chat_env=chat_env,
                          task_prompt=chat_env.env_dict['task_prompt'],
                          need_reflect=need_reflect,
                          assistant_role_name=self.assistant_role_name,
                          user_role_name=self.user_role_name,
                          phase_prompt=self.phase_prompt,
                          phase_name=self.phase_name,
                          assistant_role_prompt=self.assistant_role_prompt,
                          user_role_prompt=self.user_role_prompt,
                          chat_turn_limit=chat_turn_limit,
                          placeholders=self.phase_env,
                          memory=chat_env.memory,
                          model_type=self.model_type)
        chat_env = self.update_chat_env(chat_env)
        return chat_env


class UnitTestCodeModification(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "modality": chat_env.env_dict['modality'],
                               "language": chat_env.env_dict['language'],
                               "codes": chat_env.get_codes(),
                               "judge_triplet": chat_env.env_dict['judge_triplet']})

    def update_chat_env(self, chat_env, judge_triplet) -> ChatEnv:
        try:
            judge_triplet.unit_test_code = CodeParser.parse_code(block="", text=self.seminar_conclusion)
        except ValueError as e:
            log_visualize(f"Error parsing unit test code: {e}")
        return chat_env

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6)
    )
    def _process_single_triplet(self, chat_env, judge_triplet, unit_test_analysis, need_reflect, chat_turn_limit):
        placeholders = {
            "task": self.phase_env['task'],
            "modality": self.phase_env['modality'],
            "language": self.phase_env['language'],
            "codes": self.phase_env["codes"],
            "task_item": judge_triplet.subtask,
            "code_snippets": judge_triplet.code_snippets,
            "unit_test_code": judge_triplet.unit_test_code,
            "unit_test_analysis": unit_test_analysis
        }
        seminar_conclusion = self.chatting(
            chat_env=chat_env,
            task_prompt=chat_env.env_dict['task_prompt'],
            need_reflect=need_reflect,
            assistant_role_name=self.assistant_role_name,
            user_role_name=self.user_role_name,
            phase_prompt=self.phase_prompt,
            phase_name=self.phase_name,
            assistant_role_prompt=self.assistant_role_prompt,
            user_role_prompt=self.user_role_prompt,
            chat_turn_limit=chat_turn_limit,
            placeholders=placeholders,
            memory=chat_env.memory,
            model_type=self.model_type
        )
        return seminar_conclusion, judge_triplet

    def execute(self, chat_env, chat_turn_limit, need_reflect) -> ChatEnv:
        self.update_phase_env(chat_env)
        active_triplets = []
        for judge_triplet in self.phase_env["judge_triplet"].triplets:
            if judge_triplet.is_complete or (not judge_triplet.need_unit_test) or (not judge_triplet.unit_test_code) or judge_triplet.unit_test_result:
                continue
            conclusion = CodeParser.parse_blocks(judge_triplet.unit_test_analysis)
            if "Unit Test Code".lower() not in conclusion["Code Modification"].lower():
                continue
            unit_test_analysis = f"##Unit Test Report\n{judge_triplet.unit_test_report}\n## Analysis\n{conclusion['Analysis']}\n## Code Modification\n{conclusion['Code Modification']}\n"
            active_triplets.append((judge_triplet, unit_test_analysis))
        if not active_triplets:
            return chat_env
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_triplets)) as executor:
            futures = []
            for judge_triplet, unit_test_analysis in active_triplets:
                futures.append(executor.submit(self._process_single_triplet, chat_env, judge_triplet, unit_test_analysis, need_reflect, chat_turn_limit))
            results = concurrent.futures.wait(futures)
        for future in results.done:
            seminar_conclusion, judge_triplet = future.result()
            self.seminar_conclusion = seminar_conclusion
            chat_env = self.update_chat_env(chat_env, judge_triplet)
        return chat_env


class UnifiedCodeModification(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "modality": chat_env.env_dict['modality'],
                               "language": chat_env.env_dict['language'],
                               "codes": chat_env.get_codes(),
                               "modification_plan": chat_env.env_dict['modification_plan']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if "```".lower() in self.seminar_conclusion.lower():
            chat_env.update_codes(self.seminar_conclusion)
            chat_env.rewrite_codes("Unified Review #" + str(self.phase_env["cycle_index"]) + " Finished")
            log_visualize("**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'], self.log_filepath)))
        chat_env.env_dict['last_phase_name'] = self.phase_name
        return chat_env

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6)
    )
    def execute(self, chat_env, chat_turn_limit, need_reflect) -> ChatEnv:
        self.update_phase_env(chat_env)
        if not self.phase_env.get("modification_plan", "").strip():
            self.seminar_conclusion = ""
            chat_env = self.update_chat_env(chat_env)
            return chat_env
        self.seminar_conclusion = \
            self.chatting(chat_env=chat_env,
                          task_prompt=chat_env.env_dict['task_prompt'],
                          need_reflect=need_reflect,
                          assistant_role_name=self.assistant_role_name,
                          user_role_name=self.user_role_name,
                          phase_prompt=self.phase_prompt,
                          phase_name=self.phase_name,
                          assistant_role_prompt=self.assistant_role_prompt,
                          user_role_prompt=self.user_role_prompt,
                          chat_turn_limit=chat_turn_limit,
                          placeholders=self.phase_env,
                          memory=chat_env.memory,
                          model_type=self.model_type)
        chat_env = self.update_chat_env(chat_env)
        return chat_env


class CodeClean(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _analyze_ast_placeholders(source_code: str) -> str:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return source_code

        class PlaceholderRemover(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                node = self.generic_visit(node)
                if (not node.body or (len(node.body) == 1 and isinstance(node.body[0], ast.Pass))):
                    return None
                return node

            def visit_ClassDef(self, node):
                node = self.generic_visit(node)
                if not node.body or (len(node.body) == 1 and isinstance(node.body[0], ast.Pass)):
                    return None
                return node

        transformer = PlaceholderRemover()
        modified_tree = transformer.visit(tree)
        ast.fix_missing_locations(modified_tree)

        return ast.unparse(modified_tree)

    def update_phase_env(self, chat_env):
        self.phase_env.update({"codebooks": chat_env.codes.codebooks})

    def update_chat_env(self, chat_env):
        chat_env.rewrite_codes("Code Clean up Finished")
        log_visualize("**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'], self.log_filepath)))
        return chat_env

    def execute(self, chat_env, chat_turn_limit, need_reflect) -> ChatEnv:
        self.update_phase_env(chat_env)
        for filename, code_content in self.phase_env["codebooks"].items():
            if not code_content.strip():
                file_path = os.path.join(chat_env.env_dict['directory'], filename)
                os.remove(file_path)
                del self.phase_env["codebooks"][filename]
                continue
            self.phase_env["codebooks"][filename] = self._analyze_ast_placeholders(code_content)
        self.update_chat_env(chat_env)
        return chat_env


class DataPreparation(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        pass

    def update_chat_env(self, chat_env) -> ChatEnv:
        pass

    def execute(self, chat_env, chat_turn_limit, need_reflect) -> ChatEnv:
        target_dir = os.path.join(chat_env.env_dict['directory'], "data")
        os.makedirs(target_dir, exist_ok=True)
        shutil.copytree(chat_env.config.data_path, target_dir, dirs_exist_ok=True)
        return chat_env

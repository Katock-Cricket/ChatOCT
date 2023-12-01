import time

from revChatGPT.V3 import Chatbot

from LLM_Ref.utils import get_ref


class BaseBot:
    def start(self):
        pass

    def reset(self):
        pass

    def chat(self, message: str, ref_record: str):
        pass


class GPTBot(BaseBot):
    def __init__(self, engine: str, api_key: str, proxy: str):
        self.agent = None
        self.engine = engine
        self.api_key = api_key
        self.proxy = proxy
        self.system_prompt = "你的名字是ChatOCT，你是一个专业可靠的CAD系统，负责辅助诊断心血管相关疾病，与患者进行交流"
        self.instruction = "你作为ChatOCT智能心血管OCT辅助诊断AI医生，除非特殊情况，应当与用户使用中文沟通。"
        self.check_prompt = "用户是否在咨询任何疾病、症状、内外伤等医疗知识。如果是，请用一句话概括用户的提问并指出相关疾病的全称（例如按这样的格式回答：心绞痛，询问症状和治疗手段）；如果不是，请只回复一个数字0"

    def chat_with_gpt(self, prompt: str):
        message = ""
        it = 0
        while True:
            it += 1
            print(f"第{it}次尝试...")
            try:
                message = self.agent.ask(prompt)
            except:
                time.sleep(10)
                continue
            break
        return message

    def start(self):
        if self.agent is not None:
            self.agent.reset()
        self.agent = Chatbot(engine=self.engine, api_key=self.api_key, system_prompt=self.system_prompt,
                             proxy=self.proxy)
        res = self.chat_with_gpt(self.instruction)
        print(res)
        return

    def reset(self):
        if self.agent is None:
            self.agent.reset()

    def end(self):
        if self.agent is not None:
            self.agent = None
        else:
            print("当前没有需要停止的ChatBot")

    def chat(self, message: str, ref_record: str):
        checked_response = self.chat_with_gpt(f'{self.check_prompt}\n{message}')
        print(f"提炼用户关键词（是否有关疾病咨询）：{checked_response}")
        if checked_response != '0':
            ref = get_ref(checked_response)
            if ref is None:
                print("未在默沙东数据库中得到明确依据")
                ans = self.chat_with_gpt(f'\n{message}')
            else:
                knowledge, url = ref
                ref_prompt = f"请参考以下知识来解答用户的问题“{message}”并给出分析，请注意保持语句通顺\n[{knowledge}]"
                ans = self.chat_with_gpt(ref_prompt)
                ans += f"<br><br>参考资料来自默沙东医疗手册专业版，相关网址：{url}"
        else:
            ans = self.chat_with_gpt(f'{message}')

        print(f"历史记录：\n{ref_record}\n当前提问：\n{message}\nGPT回答：\n{ans}\n")
        return ans

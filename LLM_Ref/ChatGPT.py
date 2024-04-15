import time

from openai import OpenAI

from LLM_Ref.BaseBot import BaseBot
from LLM_Ref.utils import get_ref


class OpenAIGPTBot(BaseBot):
    def __init__(self, engine: str, api_key: str, base_url: str):
        self.agent = None
        self.engine = engine
        self.api_key = api_key
        self.base_url = base_url
        self.system_prompt = "你的名字是ChatOCT，你是一个专业可靠的CAD系统，负责辅助诊断心血管相关疾病，与患者进行交流, " \
                             "你作为ChatOCT智能心血管OCT辅助诊断AI医生，除非特殊情况，应当与用户使用中文沟通。"
        self.check_prompt = "用户是否在咨询任何疾病、症状、内外伤等医疗知识。" \
                            "如果是，请用一句话概括用户的提问并指出相关疾病的全称（例如按这样的格式回答：心绞痛，询问症状和治疗手段）；" \
                            "如果不是，请只回复一个数字0。以下是用户的问题："
        self.oct_prompt = "这是用户的心血管OCT检测报告摘要，请你总结一下检测结果，列举出可能存在的病症，给出下一步的诊疗建议"
        self.messages = []

    def check_chat(self, message):
        it = 0
        response = None
        while True:
            if it >= 5:
                raise RuntimeError("无法连接")
            it += 1
            print(f"第{it}次尝试...")
            try:
                response = self.agent.chat.completions.create(
                    model=self.engine,
                    messages=[
                        {'role': 'system', 'content': self.system_prompt},
                        {'role': 'user', 'content': message}
                    ]
                )
            except Exception as e:
                print(e)
                time.sleep(15)
                continue
            break
        ret = ''
        assert response is not None
        for choice in response.choices:
            ret += choice.message.content
        return ret

    def chat_with_gpt(self, message):
        if isinstance(message, str):
            message = {'role': 'user', 'content': message}
        self.messages.append(message)
        it = 0
        response = None
        while True:
            if it >= 5:
                raise RuntimeError("无法连接")
            it += 1
            print(f"第{it}次尝试...")
            try:
                response = self.agent.chat.completions.create(
                    model=self.engine,
                    messages=self.messages
                )
            except Exception as e:
                print(e)
                time.sleep(15)
                continue
            break
        ret = ''
        assert response is not None
        for choice in response.choices:
            ret += choice.message.content
        self.messages.append({'role': 'assistant', 'content': ret})
        # print(self.messages)
        return ret

    def start(self):
        self.agent = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        instruction = {'role': 'system', 'content': self.system_prompt}
        self.chat_with_gpt(instruction)
        return

    def ask_oct(self, abstract):
        message = {'role': 'user', 'content': f'{self.oct_prompt}\n{abstract}'}
        ans = self.chat_with_gpt(message)
        return ans

    def chat(self, message):
        if self.agent is None:
            raise RuntimeError('Chatbot not initialized')
        checked_response = self.check_chat(f'{self.check_prompt}\n{message}')
        print(f"提炼用户关键词（是否有关疾病咨询）：{checked_response}")
        time.sleep(5)
        if checked_response != '0':
            ref = get_ref(checked_response)
            if ref is None:
                print("未在默沙东数据库中得到明确依据")
                ans = self.chat_with_gpt(message)
            else:
                knowledge, url = ref
                ans = self.chat_with_gpt(
                    f"如果如下知识能够用来回答用户的问题并且与心血管疾病有关，则参考以下知识来解答用户的问题，但如果没有关联，则不能参考。这是用户的问题：" \
                    f"“{message}”\n这是给出的知识（你不一定要用到）：\n[{knowledge}]")
                ans += f"<br><br>参考资料来自默沙东医疗手册专业版，相关网址：{url}"
        else:
            ans = self.chat_with_gpt(message)
        return ans
import json
import time
from argparse import ArgumentParser

from revChatGPT.V3 import Chatbot
from openai import OpenAI
from LLM_Ref.utils import get_ref


class BaseBot:
    def start(self):
        pass

    def reset(self):
        pass

    def chat(self, message):
        pass

    def ask_oct(self, abstract):
        pass


class OpenAIGPTBot(BaseBot):
    def __init__(self, engine: str, api_key: str, proxy: str):
        self.agent = None
        self.engine = engine
        self.api_key = api_key
        self.proxy = proxy
        self.system_prompt = "你的名字是ChatOCT，你是一个专业可靠的CAD系统，负责辅助诊断心血管相关疾病，与患者进行交流, " \
                             "你作为ChatOCT智能心血管OCT辅助诊断AI医生，除非特殊情况，应当与用户使用中文沟通。"
        self.check_prompt = "用户是否在咨询任何疾病、症状、内外伤等医疗知识。" \
                            "如果是，请用一句话概括用户的提问并指出相关疾病的全称（例如按这样的格式回答：心绞痛，询问症状和治疗手段）；" \
                            "如果不是，请只回复一个数字0"
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
        print(self.messages)
        return ret

    def start(self):
        self.agent = OpenAI(api_key=self.api_key)
        instruction = {'role': 'system', 'content': self.system_prompt}
        self.chat_with_gpt(instruction)
        return

    def ask_oct(self, abstract):
        message = {'role': 'user', 'content': f'{self.oct_prompt}\n{abstract}'}
        ans = self.chat_with_gpt(message)
        return ans

    def chat(self, message):
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


class revGPTBot(OpenAIGPTBot):
    def chat_with_gpt(self, message):
        if isinstance(message, str):
            message = {'role': 'user', 'content': message}
        self.messages.append(message)
        message = ""
        it = 0
        while True:
            if it >= 5:
                raise RuntimeError("无法连接")

            it += 1
            print(f"第{it}次尝试...")
            try:
                message = self.agent.ask(self.messages)
            except Exception as e:
                print(e)
                time.sleep(15)
                continue
            break
        return message

    def start(self):
        if self.agent is not None:
            self.agent.reset()
        self.agent = Chatbot(engine=self.engine, api_key=self.api_key, system_prompt=self.system_prompt,
                             proxy=self.proxy)
        instruction = {'role': 'system', 'content': self.system_prompt}
        self.chat(instruction)
        return

    def reset(self):
        if self.agent is None:
            self.agent.reset()

    def end(self):
        if self.agent is not None:
            self.agent = None
        else:
            print("当前没有需要停止的ChatBot")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--abstract-path', default="./OCT_Det/result/2019_Jul_18_13-29-42/2019_Jul_18_13-29-42.txt",
                        required=False, help='Abstract path')
    args = parser.parse_args()

    # ChatBot = revGPTBot(
    #     engine="gpt-3.5-turbo",
    #     api_key=json.load(open('API_key.json', 'r', encoding='utf8'))['api_key'],
    #     proxy="http://127.0.0.1:7890"
    # )
    ChatBot = OpenAIGPTBot(
        engine="gpt-3.5-turbo",
        api_key=json.load(open('API_key.json', 'r', encoding='utf8'))['api_key'],
        proxy="http://127.0.0.1:7890"
    )
    ChatBot.start()
    print("ChatBot 初始化完毕")
    print("上传指定的检测摘要……")
    with open(args.abstract_path, 'r', encoding='utf8') as f:
        abstract = f.read()
    print(ChatBot.ask_oct(abstract))

    while True:
        question = input('请输入您的问题：')
        if question == 'exit':
            break
        print(ChatBot.chat(question))

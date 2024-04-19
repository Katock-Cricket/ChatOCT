import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from LLM_Ref.BaseBot import BaseBot

Huatuo_path = "LLM_Ref/Huatuo2/weights"


class Huatuo2Bot(BaseBot):
    def __init__(self):
        self.model = None
        self.engine = 'Huatuo2'
        self.tokenizer = None
        self.system_prompt = "你的名字是ChatOCT，你是一个专业可靠的心血管OCT检测系统，负责辅助诊断心血管相关疾病，与患者进行交流, " \
                             "你作为ChatOCT智能心血管OCT辅助诊断AI医生，除非特殊情况，应当与用户使用中文沟通。"
        self.check_prompt = "用户是否在咨询任何疾病、症状、内外伤等医疗知识。" \
                            "如果是，请用一句话概括用户的提问并指出相关疾病的全称（例如按这样的格式回答：心绞痛，询问症状和治疗手段）；" \
                            "如果不是，请只回复一个数字0。以下是用户的问题："
        self.oct_prompt = "这是用户的心血管OCT检测报告摘要，请你总结一下检测结果，列举出可能存在的病症，给出下一步的诊疗建议"
        self.messages = []

    def chat_with_huatuo2(self, message):
        if isinstance(message, str):
            message = {'role': 'user', 'content': message}
        self.messages.append(message)
        response = None
        try:
            response = self.model.HuatuoChat(
                tokenizer=self.tokenizer,
                messages=self.messages
            )
        except Exception as e:
            print(e)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        assert response is not None
        self.messages.append({'role': 'assistant', 'content': response})
        return response

    def start(self):
        print("init model ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            Huatuo_path,
            local_files_only=True,
            torch_dtype='auto',
            device_map="auto",
            trust_remote_code=True
        )
        self.model = self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            Huatuo_path,
            use_fast=True,
            local_files_only=True,
            trust_remote_code=True
        )
        instruction = {'role': 'system', 'content': self.system_prompt}
        print(self.chat_with_huatuo2(instruction))
        return

    def ask_oct(self, abstract):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError('Chatbot not initialized')

        self.messages = []  # 每次问OCT默认重启对话
        message = {'role': 'user', 'content': f'{self.oct_prompt}\n{abstract}'}
        ans = self.chat_with_huatuo2(message)
        return ans

    def chat(self, message):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError('Chatbot not initialized')
        ans = self.chat_with_huatuo2(message)
        return ans

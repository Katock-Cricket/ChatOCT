import json
from argparse import ArgumentParser

# from LLM_Ref.ChatGPT import OpenAIGPTBot
from LLM_Ref.DISCMed import DISCMedBot
from LLM_Ref.Huatuo2 import Huatuo2Bot

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--abstract-path', default="./OCT_Det/result/2019_Apr_22_12-53-52/2019_Apr_22_12-53-52.txt",
                        required=False, help='Abstract path')
    args = parser.parse_args()

    # ChatBot = OpenAIGPTBot(
    #     engine="gpt-3.5-turbo",
    #     api_key=json.load(open('API_key.json', 'r', encoding='utf8'))['api_key'],
    #     base_url="http://api.chatanywhere.cn/v1"
    # )
    # ChatBot = OpenAIGPTBot(
    #     engine="gpt-4-turbo",
    #     api_key=json.load(open('API_key.json', 'r', encoding='utf8'))['api_key'],
    #     base_url="http://api.chatanywhere.cn/v1"
    # )

    # ChatBot = DISCMedBot()

    ChatBot = Huatuo2Bot()

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

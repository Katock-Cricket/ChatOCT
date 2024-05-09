import json
from argparse import ArgumentParser

from LLM_Ref.BaseBot import BaseBot
from LLM_Ref.ChatGPT import OpenAIGPTBot
from LLM_Ref.DISCMed import DISCMedBot
from LLM_Ref.Huatuo2 import Huatuo2Bot
from OCT_Det.utils import NpEncoder


def eval_pipeline(ChatBot: BaseBot, abstract):
    report = ChatBot.ask_oct(abstract)
    print(report)
    first_answer = ChatBot.chat('心血管OCT检测到巨噬细胞是什么意思？')
    print(first_answer)
    single_result = {
        'report': report,
        'report_length': len(report),
        'first_answer': first_answer,
        'first_answer_length': len(first_answer),
    }
    return single_result


def read_list(fname):
    result = []
    with open(fname, "r", encoding="utf-8") as f:
        for each in f.readlines():
            each = each.strip("\n")
            with open(each, "r", encoding="utf-8") as f1:
                abstract = f1.read()
                result.append(abstract)
    return result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-e', '--engine', default='gpt3.5', type=str,
                        choices=['gpt3.5', 'gpt4', 'DISC_Med', 'Huatuo2'],
                        help='LLM engine')
    parser.add_argument('-l', '--list', default='eval/eval.lst', help='list of OCT abstract to evaluate')
    args = parser.parse_args()

    eval_list = read_list(args.list)

    ChatBot = BaseBot()
    if args.engine == 'gpt3.5':
        ChatBot = OpenAIGPTBot(
            engine='gpt-3.5-turbo',
            api_key=json.load(open('API_key.json', 'r', encoding='utf8'))['api_key_3.5'],
            base_url="https://api.chatanywhere.tech/v1",
            retrival=True
        )
    elif args.engine == 'gpt4':
        ChatBot = OpenAIGPTBot(
            engine='gpt-4-turbo',
            api_key=json.load(open('API_key.json', 'r', encoding='utf8'))['api_key_4'],
            base_url="https://api.chatanywhere.tech/v1",
            retrival=False
        )
    elif args.engine == 'DISC_Med':
        ChatBot = DISCMedBot()
    elif args.engine == 'Huatuo2':
        ChatBot = Huatuo2Bot()
    else:
        raise RuntimeError('Unknown engine')

    ChatBot.start()
    result = {
        'engine': ChatBot.engine,
        'results': [],
    }
    for abstract in eval_list:
        single_result = eval_pipeline(ChatBot, abstract)
        result['results'].append(single_result)

    with open(f'eval/report_length/result_{args.engine}.json', 'w', encoding='utf-8') as json_file:
        json_file.write('')
        json.dump(result, json_file, indent=4, ensure_ascii=False)
        print(f'saved results to result_{args.engine}.json')

import os
import torch
import platform
from colorama import Fore, Style
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


def init_model():
    print("init model ...")
    model = AutoModelForCausalLM.from_pretrained(
        "weights",
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.generation_config = GenerationConfig.from_pretrained(
        "weights"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "weights",
        use_fast=False,
        local_files_only=True,
    )
    return model, tokenizer


def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")
    print(Fore.YELLOW + Style.BRIGHT + "欢迎使用DISC-MedLLM，输入进行对话，clear 清空历史，CTRL+C 中断生成，stream 开关流式生成，exit 结束。")
    return []


def main(stream=True):
    model, tokenizer = init_model()

    messages = clear_screen()
    while True:
        prompt = input(Fore.GREEN + Style.BRIGHT + "\n用户：" + Style.NORMAL)
        if prompt.strip() == "exit":
            break
        if prompt.strip() == "clear":
            messages = clear_screen()
            continue
        print(Fore.CYAN + Style.BRIGHT + "\nDISC-MedLLM：" + Style.NORMAL, end='')
        if prompt.strip() == "stream":
            stream = not stream
            print(Fore.YELLOW + "({}流式生成)\n".format("开启" if stream else "关闭"), end='')
            continue
        messages.append({"role": "user", "content": prompt})
        if stream:
            position = 0
            try:
                for response in model.chat(tokenizer, messages, stream=True):
                    print(response[position:], end='', flush=True)
                    position = len(response)
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
            except KeyboardInterrupt:
                pass
            print()
        else:
            response = model.chat(tokenizer, messages)
            print(response)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        messages.append({"role": "assistant", "content": response})

    print(Style.RESET_ALL)


if __name__ == "__main__":
    main()
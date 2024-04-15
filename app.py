import json

import gradio as gr

from LLM_Ref.chat_bot import OpenAIGPTBot
from OCT_Det.inference_gswin import OCTDetectModel
from OCT_Det.reprocess import generate_abstract

device = "cuda:0"
api_key = json.load(open('API_key.json', 'r', encoding='utf8'))['api_key']
config = "./configs/swin/gswin_oct.py"
checkpoint = "./checkpoints/gswin_transformer.pth"
ChatBot = OpenAIGPTBot(
    engine="gpt-3.5-turbo",
    api_key=api_key,
    base_url='https://api.chatanywhere.tech/v1'
)
OCTDetector = OCTDetectModel(config, checkpoint, device)


def analyse_oct(history, file):
    OCTDetector.reset()
    OCTDetector.load_oct(file.name)
    OCTDetector.inference()
    gif_path = OCTDetector.save_results()
    abstract = generate_abstract(OCTDetector.result)
    print("Abstract: ", abstract)
    history = history + [(abstract, None)]
    return history, gr.Image.update(value=gif_path, label='OCT检测结果', loop_animation=True)


def launch_bot():
    ChatBot.start()
    print("ChatBot 初始化完毕")


def chat_oct(history):
    prompt = history[-1][0]
    response = ChatBot.ask_oct(prompt)

    history[-1][1] = response

    yield history


def analyse_abstract(history, abstract_file):
    with open(abstract_file.name, 'r', encoding='utf8') as f:
        abstract = f.read()
        history = history + [(abstract, None)]
        return history


def chat(history):
    prompt = history[-1][0]
    response = ChatBot.chat(prompt)

    history[-1][1] = response

    yield history


def add_text(history, text):
    history = history + [(text, None)]
    return history, None


def clean_data():
    return None


if __name__ == '__main__':
    with gr.Blocks(css="""
    #col_container1 {margin-left: auto; margin-right: auto;}
    #col_container2 {margin-left: auto; margin-right: auto;}
    #chatbot {height: 770px;}
    """) as demo:
        with gr.Row():
            gr.HTML("""<h1 align="center">ChatOCT 开发测试</h1>""")
        with gr.Row():
            with gr.Column(elem_id='display', scale=2):
                launch_btn = gr.Button("连接ChatGPT")
                oct_file = gr.UploadButton(file_types=[".zip"], label="上传OCT")
                abstract_file = gr.UploadButton(file_types=[".txt"], label="上传摘要")
                gif = gr.Image(label='OCT检测结果')

            with gr.Column(elem_id="col_container1", scale=8):
                chatbot = gr.Chatbot(value=[(None, "ChatOCT Demo")], label="ChatOCT", elem_id='chatbot').style(
                    height=700)
                with gr.Row():
                    with gr.Column(elem_id="col_container2", scale=6):
                        inputs = gr.Textbox(label="聊天框", placeholder="请输入文本")
                    with gr.Column(scale=1):
                        inputs_submit = gr.Button("发送")
                        clean_btn = gr.Button("清空")

        launch_btn.click(launch_bot)

        inputs.submit(add_text, [chatbot, inputs], [chatbot, inputs]).then(
            chat, [chatbot], [chatbot])

        inputs_submit.click(add_text, [chatbot, inputs], [chatbot, inputs]).then(
            chat, [chatbot], [chatbot])

        oct_file.upload(analyse_oct, [chatbot, oct_file], [chatbot, gif]).then(
            chat_oct, [chatbot], [chatbot])

        abstract_file.upload(analyse_abstract, [chatbot, abstract_file], chatbot).then(
            chat_oct, [chatbot], [chatbot])

        clean_btn.click(clean_data, [], [chatbot, inputs])
        clean_btn.click(lambda: None, None, chatbot, queue=False).success(clean_data, [], [inputs])

        demo.queue().launch(server_port=4901, server_name="127.0.0.1", favicon_path="ht.ico")

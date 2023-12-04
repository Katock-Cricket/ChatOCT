import gradio as gr
import json

from OCT_Det.inference_gswin import OCTDetectModel
from LLM_Ref.chat_bot import GPTBot
from OCT_Det.reprocess import generate_abstract

device = "cuda:0"
api_key = json.load(open('API_key.json', 'r', encoding='utf8'))['api_key']
config = "./configs/swin/gswin_oct.py"
checkpoint = "./checkpoints/gswin_transformer.pth"
ChatBot = GPTBot(
    engine="gpt-3.5-turbo",
    api_key=api_key,
    proxy="http://127.0.0.1:7890"
)
OCTDetector = OCTDetectModel(config, checkpoint, device)


def analyse_oct(history, file):
    OCTDetector.reset()
    OCTDetector.load_oct(file.name)
    OCTDetector.inference()
    OCTDetector.save_results()
    print("OCTDetector: ", OCTDetector.result)
    abstract = generate_abstract(OCTDetector.result)
    print("Abstract: ", abstract)
    history = history + [(abstract, None)]
    return history


def launch_bot():
    ChatBot.start()
    print("ChatBot 初始化完毕")


def concat_history(message_history: list) -> str:
    ret = ""
    for event in message_history:
        ret += f"{event['role']}: {event['content']}\n"
    return ret


def chat_oct(history, message_history):
    ref_record = concat_history(message_history)
    prompt = history[-1][0]
    response = ChatBot.ask_oct(prompt, ref_record)

    history[-1][1] = response
    message_history += [{'role': "user",
                         'content': prompt},
                        {'role': "assistant",
                         'content': response}]

    yield history, message_history


def chat(history, message_history):
    ref_record = concat_history(message_history)
    prompt = history[-1][0]
    response = ChatBot.chat(prompt, ref_record)

    history[-1][1] = response
    message_history += [{'role': "user",
                         'content': prompt},
                        {'role': "assistant",
                         'content': response}]

    yield history, message_history


def add_text(history, text):
    history = history + [(text, None)]
    return history, None


def clean_data():
    return [{"role": "system", "content": "ChatOCT Demo开发"}], None


if __name__ == '__main__':
    with gr.Blocks(css="""
    #col_container1 {margin-left: auto; margin-right: auto;}
    #col_container2 {margin-left: auto; margin-right: auto;}
    #chatbot {height: 770px;}
    """) as demo:
        user_history = gr.State([])

        with gr.Row():
            gr.HTML("""<h1 align="center">ChatOCT 开发测试</h1>""")
            launch_btn = gr.Button("连接ChatGPT")

        with gr.Row():
            with gr.Column(elem_id="col_container1"):
                chatbot = gr.Chatbot(value=[(None, "ChatOCT Demo")], label="ChatOCT", elem_id='chatbot').style(
                    height=700)
        with gr.Row():
            with gr.Column(elem_id="col_container2", scale=0.7):
                inputs = gr.Textbox(label="聊天框", placeholder="请输入文本")
            with gr.Column(elem_id="col_container2", scale=0.15, min_width=0, height=100):
                oct_file = gr.UploadButton(file_types=[".zip"], label="上传OCT", scale=1)
            with gr.Column(elem_id="col_container2", scale=0.15, min_width=0):
                with gr.Row():
                    inputs_submit = gr.Button("发送")
                with gr.Row():
                    clean_btn = gr.Button("清空")

        launch_btn.click(launch_bot)

        inputs.submit(add_text, [chatbot, inputs], [chatbot, inputs]).then(
            chat, [chatbot, user_history], [chatbot, user_history])

        inputs_submit.click(add_text, [chatbot, inputs], [chatbot, inputs]).then(
            chat, [chatbot, user_history], [chatbot, user_history])

        oct_file.upload(analyse_oct, [chatbot, oct_file], chatbot).then(
            chat_oct, [chatbot, user_history], [chatbot, user_history])

        clean_btn.click(clean_data, [], [chatbot, inputs])
        clean_btn.click(lambda: None, None, chatbot, queue=False).success(clean_data, [], [user_history, inputs])

        demo.queue().launch(server_port=4901, server_name="127.0.0.1", favicon_path="ht.ico")


import gradio as gr
import numpy as np

def greet(name):
    return f"Hello {name}!"

with gr.Blocks(title="测试界面") as demo:
    gr.Markdown("# 🏭 工业AI质检Copilot")
    
    with gr.Tab("测试"):
        name_input = gr.Textbox(label="输入名字")
        greet_btn = gr.Button("问候")
        greet_output = gr.Textbox(label="结果")
    
    greet_btn.click(fn=greet, inputs=name_input, outputs=greet_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7866, share=True)

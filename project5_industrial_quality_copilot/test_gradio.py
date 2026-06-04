import gradio as gr
import numpy as np

def greet(name):
    return f"Hello {name}!"

def process_image(image):
    if image is None:
        return None, "请上传图片"
    # 简单处理：转换为灰度图
    gray = np.mean(image, axis=2).astype(np.uint8)
    return gray, f"图片尺寸: {image.shape}"

with gr.Blocks(title="测试界面") as demo:
    gr.Markdown("# 🏭 工业AI质检Copilot - 测试版")
    
    with gr.Tab("测试功能"):
        name_input = gr.Textbox(label="输入名字")
        greet_btn = gr.Button("问候")
        greet_output = gr.Textbox(label="结果")
        
        image_input = gr.Image(label="上传图片")
        image_btn = gr.Button("处理图片")
        image_output = gr.Image(label="处理结果")
        image_info = gr.Textbox(label="图片信息")
    
    greet_btn.click(greet, inputs=name_input, outputs=greet_output)
    image_btn.click(process_image, inputs=image_input, outputs=[image_output, image_info])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=True
    )

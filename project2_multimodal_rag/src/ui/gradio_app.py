"""
Gradio UI for the Multimodal RAG Document Center.
"""
import gradio as gr
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")


def upload_document(file):
    """Upload a document to the server."""
    if file is None:
        return "请选择一个文件"
    
    try:
        with open(file.name, "rb") as f:
            files = {"file": (os.path.basename(file.name), f)}
            response = requests.post(f"{API_URL}/documents/upload", files=files)
        
        if response.status_code == 200:
            data = response.json()
            return f"✅ {data['message']}\n文档ID: {data['document_id']}"
        else:
            return f"❌ 上传失败: {response.text}"
    except Exception as e:
        return f"❌ 错误: {str(e)}"


def list_documents():
    """Fetch list of active documents from the server."""
    try:
        response = requests.get(f"{API_URL}/documents")
        if response.status_code == 200:
            docs = response.json()
            if not docs:
                return []
            return [(doc['title'], doc['id']) for doc in docs]
        else:
            return []
    except Exception as e:
        print(f"获取文档列表失败: {e}")
        return []


def delete_document(document_id):
    """Delete a document from the server."""
    if not document_id:
        return "请选择要删除的文档"
    
    try:
        response = requests.delete(f"{API_URL}/documents/{document_id}")
        if response.status_code == 200:
            return "✅ 删除成功"
        else:
            return f"❌ 删除失败: {response.text}"
    except Exception as e:
        return f"❌ 错误: {str(e)}"


def query_documents(query, selected_doc, use_hyde):
    """Query the RAG system with optional document filtering."""
    if not query.strip():
        return "请输入查询内容"
    
    try:
        # Prepare request payload with optional metadata filter
        payload = {"query": query, "use_hyde": use_hyde}
        
        # If a document is selected, add it as a metadata filter
        if selected_doc and selected_doc.strip():
            payload["metadata_filters"] = {"source": selected_doc}
            api_endpoint = f"{API_URL}/query-with-filter"
        else:
            api_endpoint = f"{API_URL}/query"
        
        response = requests.post(api_endpoint, json=payload)
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "暂无答案")
            sources = data.get("sources", [])
            
            if sources:
                answer += "\n\n📚 引用来源:"
                for i, source in enumerate(sources):
                    answer += f"\n{i+1}. 文档-{source.get('source', '未知')}, 页码-{source.get('page', '未知')}"
            
            return answer
        else:
            return f"❌ 查询失败: {response.text}"
    except Exception as e:
        return f"❌ 错误: {str(e)}"


def refresh_doc_list():
    """Refresh the document list dropdown."""
    docs = list_documents()
    if docs:
        return gr.update(choices=docs, value=None)
    else:
        return gr.update(choices=[("暂无文档", "")], value=None)


def on_doc_select(doc_id):
    """Handle document selection and update display."""
    doc_map = {doc_id: title for title, doc_id in list_documents()}
    if doc_id:
        title = doc_map.get(doc_id, "未找到文档")
        return f"当前分析文档: {title} (ID: {doc_id})"
    else:
        return "未选择文档（将分析所有文档）"


def create_gradio_app():
    """Create and configure the Gradio interface."""
    initial_docs = list_documents()
    initial_choices = initial_docs if initial_docs else [("暂无文档", "")]
    
    with gr.Blocks(title="多模态RAG智能文档中心", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📄 多模态RAG智能文档中心")
        
        with gr.Row():
            # Left column: Document management
            with gr.Column(scale=1):
                gr.Markdown("## 📤 文档管理")
                
                gr.Markdown("### 上传文档")
                file_input = gr.File(
                    label="选择文件 (PDF/DOCX/图片)",
                    file_types=[".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"]
                )
                upload_button = gr.Button("上传文档", variant="primary")
                upload_output = gr.Textbox(label="上传结果", lines=2)
                
                gr.Markdown("---")
                gr.Markdown("### 文档列表")
                
                doc_dropdown = gr.Dropdown(
                    label="选择文档",
                    choices=initial_choices,
                    value=None,
                    interactive=True
                )
                
                with gr.Row():
                    refresh_button = gr.Button("🔄 刷新")
                    delete_button = gr.Button("🗑️ 删除", variant="stop")
                
                delete_output = gr.Textbox(label="操作结果", lines=1)
            
            # Right column: QA interface
            with gr.Column(scale=2):
                gr.Markdown("## 💬 智能问答")
                
                with gr.Row():
                    selected_doc_display = gr.Textbox(
                        label="当前分析文档",
                        value="未选择文档（将分析所有文档）",
                        interactive=False
                    )
                
                query_input = gr.Textbox(
                    label="输入问题",
                    placeholder="请输入您的问题...",
                    lines=2
                )
                
                with gr.Row():
                    use_hyde = gr.Checkbox(label="使用HyDE优化", value=True)
                    query_button = gr.Button("提问", variant="primary", size="lg")
                
                query_output = gr.Textbox(label="回答", lines=15)
        
        # Connect buttons to functions
        upload_button.click(
            upload_document,
            inputs=[file_input],
            outputs=[upload_output]
        ).then(
            refresh_doc_list,
            outputs=[doc_dropdown]
        )
        
        refresh_button.click(
            refresh_doc_list,
            outputs=[doc_dropdown]
        )
        
        doc_dropdown.change(
            on_doc_select,
            inputs=[doc_dropdown],
            outputs=[selected_doc_display]
        )
        
        delete_button.click(
            delete_document,
            inputs=[doc_dropdown],
            outputs=[delete_output]
        ).then(
            refresh_doc_list,
            outputs=[doc_dropdown]
        ).then(
            lambda: "未选择文档（将分析所有文档）",
            outputs=[selected_doc_display]
        )
        
        query_button.click(
            query_documents,
            inputs=[query_input, doc_dropdown, use_hyde],
            outputs=[query_output]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)

"""Gradio app for the public Advanced RAG demo."""

from __future__ import annotations

import os

import gradio as gr

from src.answer import answer_question_stream
from src.document_manager import (
    delete_document,
    get_stats,
    list_documents,
    upload_document,
)


ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

CUSTOM_CSS = """
.gradio-container {
    max-width: 1000px !important;
    margin: auto !important;
}

.header-container {
    text-align: center;
    padding: 20px 0;
    margin-bottom: 10px;
}

.header-title {
    font-size: 2rem;
    font-weight: 700;
    color: #1a365d;
    margin-bottom: 8px;
}

.header-badge {
    display: inline-block;
    background: #805ad5;
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

.demo-banner {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border: 1px solid #f59e0b;
    border-radius: 8px;
    padding: 10px 16px;
    margin-bottom: 16px;
    text-align: center;
}

.demo-banner-text {
    color: #92400e;
    font-size: 0.85rem;
    margin: 0;
}
"""


def chat_stream(message: str, history: list[tuple[str, str]]):
    if not message.strip():
        yield "Please enter a question."
        return

    openai_history: list[dict[str, str]] = []
    for user_msg, assistant_msg in history:
        openai_history.append({"role": "user", "content": user_msg})
        if assistant_msg:
            openai_history.append({"role": "assistant", "content": assistant_msg})

    try:
        for partial_answer, _chunks in answer_question_stream(message, openai_history):
            yield partial_answer
    except Exception:
        yield "I apologize, but I'm having trouble processing your request right now. Please try again in a moment."


def verify_password(password: str) -> bool:
    return bool(ADMIN_PASSWORD) and password == ADMIN_PASSWORD


def handle_upload(file, doc_type):
    if file is None:
        return "Please select a file to upload.", refresh_doc_list(), refresh_stats()

    try:
        if hasattr(file, "name"):
            filename = file.name.split("/")[-1].split("\\")[-1]
            with open(file.name, "r", encoding="utf-8") as handle:
                content = handle.read()
        else:
            return "Invalid file.", refresh_doc_list(), refresh_stats()

        if not content.strip():
            return "File is empty.", refresh_doc_list(), refresh_stats()

        result = upload_document(
            text=content, filename=filename, doc_type=doc_type or "uploaded"
        )
        if result["status"] == "success":
            msg = f"Successfully uploaded '{filename}'\n"
            msg += f"Chunks created: {result['chunks_added']}\n"
            msg += f"Total chunks in KB: {result['total_documents']}"
            return msg, refresh_doc_list(), refresh_stats()

        return (
            f"Upload failed: {result['message']}",
            refresh_doc_list(),
            refresh_stats(),
        )
    except Exception as exc:
        return f"Error: {exc}", refresh_doc_list(), refresh_stats()


def handle_delete(source):
    if not source:
        return (
            "Please select a document to delete.",
            refresh_doc_list(),
            refresh_stats(),
        )

    result = delete_document(source)
    if result["status"] == "success":
        msg = f"Deleted '{source}'\n"
        msg += f"Chunks removed: {result['chunks_deleted']}\n"
        msg += f"Remaining chunks: {result['total_documents']}"
        return msg, refresh_doc_list(), refresh_stats()

    return f"Delete failed: {result['message']}", refresh_doc_list(), refresh_stats()


def refresh_doc_list():
    docs = list_documents()
    if not docs:
        return "No documents in knowledge base."

    lines = ["| Document | Type | Chunks |", "|----------|------|--------|"]
    for doc in docs:
        source = doc["source"]
        if len(source) > 50:
            source = "..." + source[-47:]
        lines.append(f"| {source} | {doc['type']} | {doc['chunk_count']} |")
    return "\n".join(lines)


def refresh_stats():
    stats = get_stats()
    return (
        f"**{stats['total_chunks']}** chunks | **{stats['total_documents']}** documents"
    )


def get_doc_choices():
    return [doc["source"] for doc in list_documents()]


def create_demo():
    with gr.Blocks(css=CUSTOM_CSS, title="Advanced RAG Demo") as demo:
        gr.HTML(
            """
            <div class="header-container">
                <div class="header-title">Advanced RAG Demo</div>
                <span class="header-badge">FICTIONAL COMPANY DATA</span>
            </div>
            """
        )

        gr.HTML(
            """
            <div class="demo-banner">
                <p class="demo-banner-text">
                    <strong>Portfolio Project</strong> - Semantic chunking, query rewriting,
                    multi-query retrieval, re-ranking, and document management.
                </p>
            </div>
            """
        )

        with gr.Tabs():
            with gr.TabItem("Chat"):
                chatbot = gr.Chatbot(
                    height=420, show_label=False, bubble_full_width=False
                )
                msg = gr.Textbox(
                    label="Ask a question",
                    placeholder="What products does Insurellm offer?",
                )
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")

                examples = gr.Examples(
                    examples=[
                        "What insurance products does Insurellm offer?",
                        "Tell me about employee Alex Chen.",
                        "What is the company culture like?",
                    ],
                    inputs=msg,
                )

                def respond(message, history):
                    history = history or []
                    history.append((message, ""))
                    for partial in chat_stream(message, history[:-1]):
                        history[-1] = (message, partial)
                        yield "", history

                send.click(fn=respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
                msg.submit(fn=respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
                clear.click(lambda: [], None, chatbot, queue=False)

            with gr.TabItem("Documents"):
                gr.Markdown("### Document Management")

                if ADMIN_PASSWORD:
                    password = gr.Textbox(label="Admin password", type="password")
                    unlock = gr.Button("Unlock")
                    access = gr.Markdown(
                        "Enter the admin password to manage documents."
                    )
                    admin_panel = gr.Column(visible=False)

                    with admin_panel:
                        file_input = gr.File(
                            label="Upload file", file_types=[".md", ".txt"]
                        )
                        doc_type = gr.Dropdown(
                            choices=[
                                "company",
                                "products",
                                "employees",
                                "contracts",
                                "uploaded",
                            ],
                            value="uploaded",
                            label="Document type",
                        )
                        upload_btn = gr.Button("Upload document", variant="primary")
                        upload_output = gr.Textbox(label="Upload result")

                        doc_list = gr.Markdown(refresh_doc_list())
                        stats = gr.Markdown(refresh_stats())

                        delete_choice = gr.Dropdown(
                            choices=get_doc_choices(),
                            label="Select a document to delete",
                        )
                        delete_btn = gr.Button("Delete document")
                        delete_output = gr.Textbox(label="Delete result")

                        upload_btn.click(
                            fn=handle_upload,
                            inputs=[file_input, doc_type],
                            outputs=[upload_output, doc_list, stats],
                        ).then(
                            fn=lambda: gr.update(choices=get_doc_choices()),
                            outputs=delete_choice,
                        )

                        delete_btn.click(
                            fn=handle_delete,
                            inputs=delete_choice,
                            outputs=[delete_output, doc_list, stats],
                        ).then(
                            fn=lambda: gr.update(choices=get_doc_choices(), value=None),
                            outputs=delete_choice,
                        )

                    def handle_unlock(input_password: str):
                        if verify_password(input_password):
                            return (
                                gr.update(value="Admin access unlocked."),
                                gr.update(visible=True),
                            )
                        return gr.update(value="Access denied."), gr.update(
                            visible=False
                        )

                    unlock.click(
                        handle_unlock, inputs=password, outputs=[access, admin_panel]
                    )
                    password.submit(
                        handle_unlock, inputs=password, outputs=[access, admin_panel]
                    )
                else:
                    gr.Markdown(
                        "Document management is disabled. Set `ADMIN_PASSWORD` in your environment to enable it."
                    )

    return demo


if __name__ == "__main__":
    create_demo().launch()

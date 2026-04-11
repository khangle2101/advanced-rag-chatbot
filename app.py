"""
AI Internal Knowledge Assistant
================================
Web interface for:
- AI Chat with knowledge base
- Document upload (incremental ingestion)
- Document management with admin authentication

Run: python app.py
URL: http://127.0.0.1:7860

"""

import os
import gradio as gr
from src.answer import answer_question
from src.document_manager import (
    upload_document,
    list_documents,
    delete_document,
    get_stats,
)

# Admin password for document management (upload/delete)
# Must be provided via environment variable before enabling admin actions.
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")


# ============ CUSTOM CSS ============
CUSTOM_CSS = """
/* Main container */
.gradio-container {
    max-width: 1000px !important;
    margin: auto !important;
}

/* Header styling */
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

/* Demo banner */
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

/* Footer */
.footer {
    text-align: center;
    padding: 15px;
    color: #718096;
    font-size: 0.85rem;
    border-top: 1px solid #e2e8f0;
    margin-top: 20px;
}

.footer-demo {
    background: #f7fafc;
    border-radius: 6px;
    padding: 10px;
    margin-top: 10px;
}
"""


# ============ CHAT FUNCTIONS ============
def chat(message: str, history: list) -> str:
    """Process user message and return AI response."""
    if not message.strip():
        return "Please enter a question."

    # Convert Gradio history format to OpenAI format
    openai_history = []
    for user_msg, assistant_msg in history:
        openai_history.append({"role": "user", "content": user_msg})
        if assistant_msg:
            openai_history.append({"role": "assistant", "content": assistant_msg})

    try:
        answer, chunks = answer_question(message, openai_history)
        return answer
    except Exception as e:
        return "I apologize, but I'm having trouble processing your request right now. Please try again in a moment."


# ============ DOCUMENT MANAGEMENT FUNCTIONS ============
def verify_password(password: str) -> bool:
    """Verify admin password."""
    return bool(ADMIN_PASSWORD) and password == ADMIN_PASSWORD


def handle_upload(file, doc_type):
    """Handle document upload."""
    if file is None:
        return "Please select a file to upload.", refresh_doc_list(), refresh_stats()

    try:
        if hasattr(file, "name"):
            filename = file.name.split("/")[-1].split("\\")[-1]
            with open(file.name, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            return "Invalid file.", refresh_doc_list(), refresh_stats()

        if not content.strip():
            return "File is empty.", refresh_doc_list(), refresh_stats()

        result = upload_document(
            text=content,
            filename=filename,
            doc_type=doc_type or "uploaded",
        )

        if result["status"] == "success":
            msg = f"Successfully uploaded '{filename}'\n"
            msg += f"Chunks created: {result['chunks_added']}\n"
            msg += f"Total documents in KB: {result['total_documents']}"
            return msg, refresh_doc_list(), refresh_stats()
        else:
            return (
                f"Upload failed: {result['message']}",
                refresh_doc_list(),
                refresh_stats(),
            )

    except Exception as e:
        return f"Error: {str(e)}", refresh_doc_list(), refresh_stats()


def handle_delete(source):
    """Handle document deletion."""
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
        msg += f"Remaining documents: {result['total_documents']}"
        return msg, refresh_doc_list(), refresh_stats()
    else:
        return (
            f"Delete failed: {result['message']}",
            refresh_doc_list(),
            refresh_stats(),
        )


def refresh_doc_list():
    """Refresh the document list."""
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
    """Refresh statistics display."""
    stats = get_stats()
    return (
        f"**{stats['total_chunks']}** chunks | **{stats['total_documents']}** documents"
    )


def get_doc_choices():
    """Get list of documents for dropdown."""
    docs = list_documents()
    return [doc["source"] for doc in docs]


# ============ CREATE INTERFACE ============
def create_demo():
    """Create the Gradio interface with tabs."""

    with gr.Blocks(css=CUSTOM_CSS, title="AI Internal Knowledge Assistant") as demo:
        # Header
        gr.HTML("""
            <div class="header-container">
                <div class="header-title">AI Internal Knowledge Assistant</div>
                <span class="header-badge">INTERNAL USE ONLY</span>
            </div>
        """)

        # Demo disclaimer
        gr.HTML("""
            <div class="demo-banner">
                <p class="demo-banner-text">
                    <strong>Demo with Fictional Data</strong> - 
                    Portfolio project showcasing RAG technology. All data is fictional.
                </p>
            </div>
        """)

        # Tabs
        with gr.Tabs():
            # ============ TAB 1: CHAT ============
            with gr.TabItem("Chat"):
                chatbot = gr.Chatbot(
                    height=400,
                    show_label=False,
                    bubble_full_width=False,
                )

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask about products, employees, contracts, policies...",
                        show_label=False,
                        container=False,
                        scale=9,
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", size="sm")

                gr.Markdown("### Quick Questions")
                gr.Examples(
                    examples=[
                        ["What products does the company offer?"],
                        ["Tell me about employee benefits"],
                        ["What is the company culture?"],
                    ],
                    inputs=msg,
                    label="",
                )

                # Chat event handlers
                def respond(message, chat_history):
                    if not message.strip():
                        return "", chat_history
                    bot_message = chat(message, chat_history)
                    chat_history.append((message, bot_message))
                    return "", chat_history

                def clear_chat_fn():
                    return [], ""

                msg.submit(respond, [msg, chatbot], [msg, chatbot])
                submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
                clear_btn.click(clear_chat_fn, outputs=[chatbot, msg])

            # ============ TAB 2: DOCUMENT MANAGEMENT ============
            with gr.TabItem("Documents"):
                # ===== LOGIN SECTION =====
                with gr.Column(visible=True) as login_section:
                    gr.Markdown("### Admin Authentication Required")
                    gr.Markdown(
                        "This section contains sensitive document management functions. "
                        "Set the `ADMIN_PASSWORD` environment variable, then enter it here to continue."
                    )

                    with gr.Row():
                        admin_password = gr.Textbox(
                            label="Admin Password",
                            type="password",
                            placeholder="Enter password...",
                            scale=3,
                        )
                        unlock_btn = gr.Button("Unlock", variant="primary", scale=1)

                    login_status = gr.Markdown("")

                # ===== ADMIN PANEL =====
                with gr.Column(visible=False) as admin_panel:
                    gr.Markdown("### Knowledge Base Statistics")
                    stats_display = gr.Markdown(value=refresh_stats())

                    gr.Markdown("---")

                    gr.Markdown("### Upload New Document")
                    gr.Markdown(
                        "*Upload `.md` or `.txt` files. Documents are automatically chunked and indexed.*"
                    )

                    with gr.Row():
                        file_input = gr.File(
                            label="Select File",
                            file_types=[".md", ".txt"],
                            file_count="single",
                        )
                        doc_type_input = gr.Dropdown(
                            choices=[
                                "company",
                                "products",
                                "employees",
                                "contracts",
                                "policies",
                                "other",
                            ],
                            value="other",
                            label="Document Type",
                        )

                    upload_btn = gr.Button("Upload Document", variant="primary")
                    upload_status = gr.Textbox(
                        label="Status", interactive=False, lines=3
                    )

                    gr.Markdown("---")

                    gr.Markdown("### Documents in Knowledge Base")
                    doc_list_display = gr.Markdown(value=refresh_doc_list())

                    refresh_btn = gr.Button("Refresh List", size="sm")

                    gr.Markdown("---")

                    gr.Markdown("### Delete Document")
                    gr.Markdown(
                        "*Select a document to remove from the knowledge base.*"
                    )

                    with gr.Row():
                        delete_dropdown = gr.Dropdown(
                            choices=get_doc_choices(),
                            label="Select Document",
                            interactive=True,
                        )
                        delete_btn = gr.Button("Delete", variant="stop")

                    delete_status = gr.Textbox(
                        label="Status", interactive=False, lines=2
                    )

                # ===== EVENT HANDLERS =====
                def handle_unlock(password):
                    if not ADMIN_PASSWORD:
                        return (
                            gr.Column(visible=True),
                            gr.Column(visible=False),
                            "**Admin access is disabled.** Set `ADMIN_PASSWORD` in your environment to enable document management.",
                            gr.update(),
                            gr.update(),
                            gr.update(),
                        )
                    if verify_password(password):
                        return (
                            gr.Column(visible=False),
                            gr.Column(visible=True),
                            "",
                            refresh_stats(),
                            refresh_doc_list(),
                            gr.Dropdown(choices=get_doc_choices()),
                        )
                    else:
                        return (
                            gr.Column(visible=True),
                            gr.Column(visible=False),
                            "**Access denied.** Incorrect password.",
                            gr.update(),
                            gr.update(),
                            gr.update(),
                        )

                unlock_btn.click(
                    handle_unlock,
                    inputs=[admin_password],
                    outputs=[
                        login_section,
                        admin_panel,
                        login_status,
                        stats_display,
                        doc_list_display,
                        delete_dropdown,
                    ],
                )

                admin_password.submit(
                    handle_unlock,
                    inputs=[admin_password],
                    outputs=[
                        login_section,
                        admin_panel,
                        login_status,
                        stats_display,
                        doc_list_display,
                        delete_dropdown,
                    ],
                )

                upload_btn.click(
                    handle_upload,
                    inputs=[file_input, doc_type_input],
                    outputs=[upload_status, doc_list_display, stats_display],
                )

                def refresh_all():
                    return (
                        refresh_doc_list(),
                        refresh_stats(),
                        gr.Dropdown(choices=get_doc_choices()),
                    )

                refresh_btn.click(
                    refresh_all,
                    outputs=[doc_list_display, stats_display, delete_dropdown],
                )

                def delete_and_refresh(source):
                    status, doc_list, stats = handle_delete(source)
                    return (
                        status,
                        doc_list,
                        stats,
                        gr.Dropdown(choices=get_doc_choices()),
                    )

                delete_btn.click(
                    delete_and_refresh,
                    inputs=[delete_dropdown],
                    outputs=[
                        delete_status,
                        doc_list_display,
                        stats_display,
                        delete_dropdown,
                    ],
                )

        # Footer
        gr.HTML("""
            <div class="footer">
                <p><strong>AI Internal Knowledge Assistant</strong></p>
                <div class="footer-demo">
                    <p style="font-size: 0.75rem; color: #718096; margin: 0;">
                        <strong>Portfolio Project</strong> |
                        Built with RAG, ChromaDB & Gradio
                    </p>
                </div>
            </div>
        """)

    return demo


# Create and launch
demo = create_demo()


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  AI INTERNAL KNOWLEDGE ASSISTANT")
    print("  Portfolio Project - RAG Demo")
    print("=" * 50)
    print("\n  Features:")
    print("  - Chat: Ask questions about knowledge base")
    print("  - Documents: Upload/manage documents (requires admin password)")
    print(f"\n  URL: http://127.0.0.1:7860\n")
    print("=" * 50 + "\n")

    demo.launch(
        server_port=7860,
        share=False,
        show_error=False,
    )

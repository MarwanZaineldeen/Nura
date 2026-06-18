from __future__ import annotations

import html
import os
from functools import lru_cache

import gradio as gr
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

try:
    from src.retrieval.retrieval_engine import RetrievalEngine
except ModuleNotFoundError:
    from retrieval_engine import RetrievalEngine


THEME = gr.themes.Base(
    primary_hue="teal",
    secondary_hue="rose",
    neutral_hue="slate",
    radius_size="sm",
)

CSS = """
.retrieval-shell {
    max-width: 1120px;
    margin: 0 auto;
}
.retrieval-header {
    background: #0f172a;
    color: #f8fafc;
    padding: 20px;
    border-left: 8px solid #14b8a6;
    margin-bottom: 18px;
}
.retrieval-header h1 {
    margin: 4px 0 0;
}
.retrieval-kicker {
    color: #5eead4;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 0;
    text-transform: uppercase;
}
.retrieval-panel {
    border: 2px solid #0f172a;
    background: #ffffff;
    padding: 16px;
}
.result-card {
    border: 2px solid #0f172a;
    background: #f8fafc;
    padding: 14px;
    margin-bottom: 12px;
}
.result-topline {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: center;
    margin-bottom: 8px;
}
.result-title {
    font-weight: 800;
    color: #0f172a;
}
.result-score {
    background: #14b8a6;
    color: #042f2e;
    font-weight: 800;
    padding: 4px 8px;
    border: 1px solid #0f172a;
    white-space: nowrap;
}
.source-pill {
    display: inline-block;
    border: 1px solid #0f172a;
    padding: 3px 7px;
    margin-right: 6px;
    font-size: 12px;
    font-weight: 700;
    background: #ffe4e6;
}
.result-text {
    color: #334155;
    line-height: 1.48;
}
.retrieval-error {
    border: 2px solid #be123c;
    background: #fff1f2;
    padding: 14px;
}
"""


@lru_cache(maxsize=1)
def get_engine() -> RetrievalEngine:
    return RetrievalEngine()


def _shorten(text: str, limit: int = 900) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def _result_html(results: list[dict]) -> str:
    if not results:
        return "<div class='retrieval-error'>No retrieval results were returned.</div>"

    cards = []
    for result in results:
        title = html.escape(str(result["title"]))
        topic = html.escape(str(result["topic"]))
        source_type = html.escape(str(result["source_type"]).upper())
        record_id = html.escape(str(result["id"]))
        text = html.escape(_shorten(str(result["text"]))).replace("\n", "<br>")
        score = float(result["score"])

        cards.append(
            "<div class='result-card'>"
            "<div class='result-topline'>"
            f"<div class='result-title'>#{result['rank']} - {title}</div>"
            f"<div class='result-score'>{score:.4f}</div>"
            "</div>"
            f"<div><span class='source-pill'>{source_type}</span><span class='source-pill'>{topic}</span></div>"
            f"<p class='result-text'>{text}</p>"
            f"<div class='result-text'><b>ID:</b> {record_id}</div>"
            "</div>"
        )

    return "".join(cards)


def search(query: str, source: str, top_k: int) -> tuple[str, list[list[str]]]:
    clean_query = (query or "").strip()
    if not clean_query:
        return "<div class='retrieval-error'>Please enter a query to search.</div>", []

    try:
        results = get_engine().search(clean_query, source=source, top_k=int(top_k))
    except EnvironmentError:
        return (
            "<div class='retrieval-error'>Qdrant settings are missing. Set QDRANT_URL and QDRANT_API_KEY in .env.</div>",
            [],
        )
    except (ResponseHandlingException, UnexpectedResponse):
        return (
            "<div class='retrieval-error'>Qdrant search failed. Check your internet connection and Qdrant cluster status.</div>",
            [],
        )
    except Exception as exc:
        print(f"Retrieval UI error: {type(exc).__name__}: {exc}")
        return "<div class='retrieval-error'>Retrieval is unavailable right now. Check the terminal logs.</div>", []

    details = [
        ["Query", clean_query],
        ["Source Filter", source],
        ["Returned Results", str(len(results))],
        ["Top Score", f"{results[0]['score']:.4f}" if results else "N/A"],
    ]
    return _result_html(results), details


with gr.Blocks(title="RAG Retrieval Tester") as interface:
    with gr.Column(elem_classes=["retrieval-shell"]):
        gr.HTML(
            """
            <div class="retrieval-header">
                <div class="retrieval-kicker">Module 4</div>
                <h1>RAG Retrieval Tester</h1>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=4, elem_classes=["retrieval-panel"]):
                query_input = gr.Textbox(
                    lines=6,
                    label="User query",
                    placeholder="Example: I feel anxious and cannot sleep.",
                )
                with gr.Row():
                    source_input = gr.Radio(
                        choices=["both", "cci", "amod"],
                        value="both",
                        label="Retrieval source",
                    )
                    top_k_input = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Top K",
                    )
                search_button = gr.Button("Search knowledge base", variant="primary")

            with gr.Column(scale=5):
                result_output = gr.HTML()
                details_output = gr.Dataframe(
                    headers=["Field", "Value"],
                    datatype=["str", "str"],
                    label="Search Details",
                    interactive=False,
                )

        search_button.click(
            fn=search,
            inputs=[query_input, source_input, top_k_input],
            outputs=[result_output, details_output],
        )


if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", "7864"))
    interface.launch(
        theme=THEME,
        css=CSS,
        server_name="127.0.0.1",
        server_port=port,
        prevent_thread_lock=False,
    )

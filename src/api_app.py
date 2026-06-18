from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "src" / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.append(str(MODELS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from chatbot_pipeline import ChatbotPipeline


app = FastAPI(
    title="Mental Health Support Chatbot",
    description="Integrated language, emotion, intent, RAG, guardrail, and response-generation API.",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    source: str = Field("both", pattern="^(both|cci|amod)$")
    top_k: int = Field(8, ge=1, le=10)
    history: list[dict[str, str]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: str
    state: dict[str, Any]


@lru_cache(maxsize=1)
def get_pipeline() -> ChatbotPipeline:
    return ChatbotPipeline()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    pipeline = get_pipeline()
    pipeline.retrieval_source = request.source
    pipeline.top_k = request.top_k
    output = pipeline.run(request.message, history=request.history)
    return ChatResponse(response=output["response"], state=output["state"])


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PRODUCTION_PAGE


@app.get("/developer", response_class=HTMLResponse)
def developer() -> str:
    return DEVELOPER_PAGE


PRODUCTION_PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mental Health Support Chatbot</title>
  <style>
    :root {
      --bg: #f6f7fb;
      --panel: #ffffff;
      --ink: #121826;
      --muted: #667085;
      --line: #d9dee8;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --soft: #e7f7f3;
      --user: #0f766e;
      --assistant: #ffffff;
      --shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    .layout {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 320px 1fr;
    }
    aside {
      background: #101828;
      color: white;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 22px;
    }
    .brand span {
      color: #5eead4;
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
    }
    .brand h1 {
      margin: 6px 0 0;
      font-size: 28px;
      line-height: 1.08;
      letter-spacing: 0;
    }
    .mode-group {
      display: grid;
      gap: 10px;
    }
    .mode {
      border: 1px solid rgba(255,255,255,0.18);
      background: rgba(255,255,255,0.06);
      color: white;
      padding: 12px;
      text-align: left;
      cursor: pointer;
    }
    .mode.active {
      background: var(--soft);
      color: #064e3b;
      border-color: #99f6e4;
    }
    .mode b { display: block; margin-bottom: 3px; }
    .mode span { color: inherit; opacity: 0.76; font-size: 13px; }
    .side-note {
      color: #cbd5e1;
      font-size: 13px;
      line-height: 1.5;
      margin-top: auto;
    }
    .privacy-note {
      border: 1px solid rgba(255,255,255,0.16);
      background: rgba(255,255,255,0.06);
      padding: 12px;
      color: #e2e8f0;
      font-size: 13px;
      line-height: 1.45;
    }
    main {
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-width: 0;
    }
    .topbar {
      padding: 18px 24px;
      background: rgba(255,255,255,0.82);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(12px);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    .topbar b { display: block; }
    .topbar span { color: var(--muted); font-size: 13px; }
    .clear {
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      padding: 9px 12px;
      font-weight: 750;
      cursor: pointer;
    }
    .chat {
      padding: 24px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .bubble {
      max-width: min(760px, 88%);
      padding: 14px 16px;
      line-height: 1.55;
      white-space: pre-wrap;
      box-shadow: var(--shadow);
      border-radius: 18px;
    }
    .bubble.user {
      align-self: flex-end;
      background: var(--user);
      color: white;
      border-bottom-right-radius: 4px;
    }
    .bubble.assistant {
      align-self: flex-start;
      background: var(--assistant);
      border: 1px solid var(--line);
      border-bottom-left-radius: 4px;
    }
    .typing {
      display: inline-flex;
      gap: 5px;
      align-items: center;
      min-width: 58px;
    }
    .typing span {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: #94a3b8;
      animation: bounce 1.15s infinite ease-in-out;
    }
    .typing span:nth-child(2) { animation-delay: 0.15s; }
    .typing span:nth-child(3) { animation-delay: 0.3s; }
    @keyframes bounce {
      0%, 80%, 100% { transform: translateY(0); opacity: 0.45; }
      40% { transform: translateY(-5px); opacity: 1; }
    }
    .composer {
      padding: 18px 24px 24px;
      border-top: 1px solid var(--line);
      background: rgba(246,247,251,0.96);
    }
    .composer-inner {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      max-width: 980px;
      margin: 0 auto;
    }
    textarea {
      width: 100%;
      min-height: 58px;
      max-height: 170px;
      resize: vertical;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      padding: 13px 14px;
      font: inherit;
      line-height: 1.45;
      box-shadow: var(--shadow);
    }
    .send {
      border: 1px solid var(--accent-dark);
      background: var(--accent);
      color: white;
      padding: 0 22px;
      min-width: 112px;
      font: inherit;
      font-weight: 850;
      cursor: pointer;
      box-shadow: var(--shadow);
    }
    .send:disabled { opacity: 0.65; cursor: wait; }
    @media (max-width: 860px) {
      .layout { grid-template-columns: 1fr; }
      aside { min-height: auto; }
      .composer-inner { grid-template-columns: 1fr; }
      .send { min-height: 46px; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside>
      <div class="brand">
        <span>Support Chat</span>
        <h1>Mental Health Assistant</h1>
      </div>
      <div class="mode-group">
        <button class="mode active" data-source="both"><b>Balanced Support</b><span>Uses educational sheets and counseling examples.</span></button>
        <button class="mode" data-source="cci"><b>Educational Guidance</b><span>Grounded in structured self-help information sheets.</span></button>
        <button class="mode" data-source="amod"><b>Counseling Style</b><span>Uses similar counseling Q&A examples.</span></button>
      </div>
      <div class="side-note">The assistant keeps a short memory of the latest conversation turns and uses 8 retrieved passages when RAG is needed.</div>
      <div class="privacy-note">Short-term memory stays in this browser session and helps the assistant keep context without showing technical details.</div>
    </aside>
    <main>
      <div class="topbar">
        <div><b>Conversation</b><span>Supportive responses with safety guardrails</span></div>
        <button class="clear" id="clear">New chat</button>
      </div>
      <div class="chat" id="chat">
        <div class="bubble assistant">Hi, I am here with you. Share what is on your mind, and I will try to respond supportively.</div>
      </div>
      <div class="composer">
        <div class="composer-inner">
          <textarea id="message" placeholder="Write your message..."></textarea>
          <button class="send" id="send">Send</button>
        </div>
      </div>
    </main>
  </div>
  <script>
    const chat = document.getElementById("chat");
    const message = document.getElementById("message");
    const send = document.getElementById("send");
    const clear = document.getElementById("clear");
    const modeButtons = [...document.querySelectorAll(".mode")];
    let source = "both";
    let history = [];

    function addBubble(role, text) {
      const bubble = document.createElement("div");
      bubble.className = `bubble ${role}`;
      bubble.textContent = text;
      chat.appendChild(bubble);
      chat.scrollTop = chat.scrollHeight;
      return bubble;
    }

    function addTypingBubble() {
      const bubble = document.createElement("div");
      bubble.className = "bubble assistant";
      bubble.innerHTML = "<div class='typing'><span></span><span></span><span></span></div>";
      chat.appendChild(bubble);
      chat.scrollTop = chat.scrollHeight;
      return bubble;
    }

    modeButtons.forEach((button) => {
      button.addEventListener("click", () => {
        modeButtons.forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        source = button.dataset.source;
      });
    });

    clear.addEventListener("click", () => {
      history = [];
      chat.innerHTML = "";
      addBubble("assistant", "New chat started. What would you like to talk about?");
    });

    async function submitMessage() {
      const text = message.value.trim();
      if (!text) return;

      addBubble("user", text);
      history.push({ role: "user", content: text });
      history = history.slice(-10);
      message.value = "";
      send.disabled = true;
      send.textContent = "...";
      const typingBubble = addTypingBubble();

      try {
        const response = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, source, top_k: 8, history }),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        typingBubble.remove();
        addBubble("assistant", data.response || "I am here with you, but I could not generate a full response. Could you tell me a little more?");
        history.push({ role: "assistant", content: data.response || "" });
        history = history.slice(-10);
      } catch (error) {
        typingBubble.remove();
        addBubble("assistant", "I had trouble responding just now. Please try again in a moment.");
      } finally {
        send.disabled = false;
        send.textContent = "Send";
      }
    }

    send.addEventListener("click", submitMessage);
    message.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        submitMessage();
      }
    });
  </script>
</body>
</html>
"""


DEVELOPER_PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mental Health Support Chatbot</title>
  <style>
    :root {
      --ink: #111827;
      --muted: #4b5563;
      --line: #d1d5db;
      --paper: #f8fafc;
      --panel: #ffffff;
      --green: #059669;
      --green-soft: #d1fae5;
      --indigo: #4338ca;
      --rose: #be123c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--paper);
    }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    header {
      background: #0f172a;
      color: white;
      border-bottom: 4px solid var(--green);
    }
    .header-inner {
      max-width: 1180px;
      margin: 0 auto;
      padding: 18px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }
    .brand {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .brand span {
      color: #86efac;
      font-size: 12px;
      text-transform: uppercase;
      font-weight: 800;
    }
    .brand h1 {
      margin: 0;
      font-size: clamp(22px, 3vw, 32px);
      letter-spacing: 0;
    }
    main {
      max-width: 1180px;
      width: 100%;
      margin: 0 auto;
      padding: 20px;
      display: grid;
      grid-template-columns: minmax(300px, 420px) 1fr;
      gap: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 16px;
    }
    label {
      display: block;
      font-weight: 750;
      margin-bottom: 8px;
    }
    textarea, select, input {
      width: 100%;
      border: 1px solid var(--line);
      color: var(--ink);
      background: white;
      padding: 10px 11px;
      font: inherit;
    }
    textarea {
      min-height: 190px;
      resize: vertical;
      line-height: 1.45;
    }
    .controls {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin: 14px 0;
    }
    button {
      width: 100%;
      border: 1px solid #064e3b;
      background: var(--green);
      color: white;
      padding: 11px 14px;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
    }
    button:disabled {
      opacity: 0.65;
      cursor: wait;
    }
    .answer {
      min-height: 190px;
      line-height: 1.58;
      white-space: pre-wrap;
    }
    .route {
      display: inline-block;
      background: var(--green-soft);
      border: 1px solid #065f46;
      color: #064e3b;
      padding: 4px 8px;
      font-size: 12px;
      font-weight: 850;
      margin-bottom: 12px;
      text-transform: uppercase;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-top: 14px;
    }
    .metric {
      border: 1px solid var(--line);
      background: #fafafa;
      padding: 12px;
      min-height: 72px;
    }
    .metric b {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }
    .metric span {
      font-weight: 800;
    }
    details {
      margin-top: 14px;
      border: 1px solid var(--line);
      background: #fafafa;
      padding: 10px;
    }
    summary {
      cursor: pointer;
      font-weight: 800;
    }
    pre {
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      color: #1f2937;
    }
    .error {
      color: var(--rose);
      font-weight: 750;
    }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      .grid { grid-template-columns: 1fr; }
      .controls { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="header-inner">
        <div class="brand">
          <span>Integrated RAG Pipeline</span>
          <h1>Mental Health Support Chatbot</h1>
        </div>
      </div>
    </header>

    <main>
      <section class="panel">
        <label for="message">User message</label>
        <textarea id="message" placeholder="Example: I feel anxious every night and cannot sleep."></textarea>

        <div class="controls">
          <div>
            <label for="source">Retrieval mode</label>
            <select id="source">
              <option value="both">Balanced Support</option>
              <option value="cci">Educational Guidance</option>
              <option value="amod">Counseling Style</option>
            </select>
          </div>
          <div>
            <label for="topK">Retrieved chunks</label>
            <input id="topK" type="number" min="1" max="10" value="5" />
          </div>
        </div>

        <button id="send">Generate Response</button>
      </section>

      <section class="panel">
        <div id="route" class="route">Waiting</div>
        <div id="answer" class="answer">Enter a message to run the full pipeline.</div>

        <div class="grid">
          <div class="metric"><b>Language</b><span id="language">-</span></div>
          <div class="metric"><b>Emotion</b><span id="emotion">-</span></div>
          <div class="metric"><b>Intent</b><span id="intent">-</span></div>
        </div>

        <details>
          <summary>Pipeline State</summary>
          <pre id="state">{}</pre>
        </details>
      </section>
    </main>
  </div>

  <script>
    const sendButton = document.getElementById("send");
    const answer = document.getElementById("answer");
    const stateBox = document.getElementById("state");
    const route = document.getElementById("route");
    const language = document.getElementById("language");
    const emotion = document.getElementById("emotion");
    const intent = document.getElementById("intent");

    function pct(value) {
      if (typeof value !== "number") return "";
      return ` (${Math.round(value * 100)}%)`;
    }

    sendButton.addEventListener("click", async () => {
      const message = document.getElementById("message").value.trim();
      const source = document.getElementById("source").value;
      const topK = Number(document.getElementById("topK").value || 5);

      if (!message) {
        answer.innerHTML = "<span class='error'>Please enter a message.</span>";
        return;
      }

      sendButton.disabled = true;
      sendButton.textContent = "Running pipeline...";
      answer.textContent = "Analyzing language, emotion, intent, safety, retrieval, and generation...";

      try {
        const response = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message, source, top_k: topK }),
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        const state = data.state || {};

        answer.textContent = data.response || "";
        route.textContent = state.route || "unknown";
        language.textContent = `${state.language?.language_name || "-"}${pct(state.language?.confidence)}`;
        emotion.textContent = `${state.emotion?.emotion || "-"}${pct(state.emotion?.confidence)}`;
        intent.textContent = `${state.intent?.intent || "-"}${pct(state.intent?.confidence)}`;
        stateBox.textContent = JSON.stringify(state, null, 2);
      } catch (error) {
        answer.innerHTML = `<span class='error'>Request failed: ${error.message}</span>`;
      } finally {
        sendButton.disabled = false;
        sendButton.textContent = "Generate Response";
      }
    });
  </script>
</body>
</html>
"""

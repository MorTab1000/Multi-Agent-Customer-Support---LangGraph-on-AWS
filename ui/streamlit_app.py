import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import re
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

DEFAULT_SUPPORT_API_URL = "https://46r8ga4hcc.us-east-1.awsapprunner.com/ask"
REQUEST_TIMEOUT_SECONDS = 30


def sanitize_latex(text: str) -> str:
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    return text


def resolve_api_url() -> str:
    """Resolve API URL from env first (local dev), then secrets (cloud), then fallback."""
    env_url = os.getenv("SUPPORT_API_URL")
    if env_url:
        return env_url

    try:
        secrets_url = st.secrets.get("SUPPORT_API_URL")
        if secrets_url:
            return secrets_url
    except Exception:
        pass

    return DEFAULT_SUPPORT_API_URL


def fetch_answer(api_url: str, question: str) -> dict[str, Any]:
    payload = json.dumps({"question": question}).encode("utf-8")
    request = Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            raw_data = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Support service returned HTTP {exc.code}. Details: {detail[:240]}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            "Support service is currently unavailable. Please try again in a moment."
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError("Request timed out while contacting support service.") from exc

    try:
        data = json.loads(raw_data)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Support service returned invalid JSON.") from exc

    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Support service response is missing a valid answer.")

    confidence_raw = data.get("confidence")
    confidence = None
    if isinstance(confidence_raw, (int, float)):
        confidence = max(0.0, min(1.0, float(confidence_raw)))

    escalated = bool(data.get("escalated", False))
    return {"answer": answer, "confidence": confidence, "escalated": escalated}


def render_assistant_message(content: str, confidence: float | None, escalated: bool) -> None:
    sanitized_content = sanitize_latex(content)
    avatar = "🧑‍🏫" if escalated else "🤖"
    with st.chat_message("assistant", avatar=avatar):
        if escalated:
            st.info("Escalated to TA for review")
            st.markdown(sanitized_content)
        else:
            st.markdown(sanitized_content)

        if confidence is None:
            st.caption("Confidence Score: N/A")
        else:
            st.caption(f"Confidence Score: {confidence:.2f}")


st.set_page_config(
    page_title="Multi-Agent Accademic Assistant",
    page_icon="💬",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 980px;}
      [data-testid="stChatMessage"] p {line-height: 1.55;}
      [data-testid="stCaptionContainer"] {margin-top: 0.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Multi-Multi-Agent Accademic Assistant")
st.caption("Ask questions and get AI or TA-escalated responses.")

api_url = resolve_api_url()
with st.sidebar:
    st.subheader("Configuration")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="🙋"):
            st.markdown(message["content"])
    else:
        render_assistant_message(
            content=message["content"],
            confidence=message.get("confidence"),
            escalated=bool(message.get("escalated", False)),
        )

prompt = st.chat_input("Ask your question...")
if prompt is not None:
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        st.warning("Please enter a question before sending.")
    else:
        user_message = {"role": "user", "content": cleaned_prompt}
        st.session_state.messages.append(user_message)
        with st.chat_message("user", avatar="🙋"):
            st.markdown(cleaned_prompt)

        with st.spinner("Thinking..."):
            try:
                result = fetch_answer(api_url, cleaned_prompt)
            except RuntimeError as exc:
                result = {
                    "answer": f"Unable to process your request right now.\n\n{exc}",
                    "confidence": None,
                    "escalated": False,
                }

        assistant_message = {
            "role": "assistant",
            "content": result["answer"],
            "confidence": result["confidence"],
            "escalated": result["escalated"],
        }
        st.session_state.messages.append(assistant_message)
        render_assistant_message(
            content=assistant_message["content"],
            confidence=assistant_message["confidence"],
            escalated=assistant_message["escalated"],
        )

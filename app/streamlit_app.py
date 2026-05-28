"""Streamlit UI for the Legal RAG bot."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.chains.rag_chain import get_hr_bot


@st.cache_resource
def load_bot():
    return get_hr_bot()


def main() -> None:
    st.set_page_config(page_title="Labor Law Bot", page_icon="🤖")
    st.title("Chat bot luật Lao động Việt Nam")
    st.markdown("Xin chào! Tôi có thể giúp gì cho bạn?")

    rag_chain = load_bot()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Nhập câu hỏi..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ và tra cứu..."):
                try:
                    response = rag_chain.invoke(
                        {
                            "input": prompt,
                            "chat_history": st.session_state.chat_history,
                        }
                    )

                    bot_reply = response["answer"]
                    context_used = response["context"]

                    st.markdown(bot_reply)

                    with st.expander("📄 Xem tài liệu trích xuất"):
                        if context_used.strip() == "":
                            st.warning("Không tìm thấy ngữ cảnh nào phù hợp trong PDF.")
                        else:
                            st.text_area(
                                "Văn bản AI đã đọc:",
                                value=context_used,
                                height=250,
                            )

                    st.session_state.messages.append(
                        {"role": "assistant", "content": bot_reply}
                    )
                    st.session_state.chat_history.extend(
                        [
                            HumanMessage(content=prompt),
                            AIMessage(content=bot_reply),
                        ]
                    )
                except Exception as exc:
                    st.error(f"Có lỗi xảy ra: {exc}")


if __name__ == "__main__":
    main()

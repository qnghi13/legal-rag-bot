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
    return get_hr_bot(return_context_list=True)


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
                    dq_context = response.get("dq_context", "")
                    gq_context = response.get("gq_context", "")

                    st.markdown(bot_reply)

                    with st.expander("📄 Xem ngữ cảnh truy xuất"):
                        if not dq_context.strip() and not gq_context.strip():
                            st.warning("Không tìm thấy ngữ cảnh nào phù hợp.")
                        if dq_context.strip():
                            st.text_area(
                                "Dq - tài liệu truy xuất:",
                                value=dq_context,
                                height=250,
                            )
                        if gq_context.strip():
                            st.text_area(
                                "Gq - quan hệ từ graph:",
                                value=gq_context,
                                height=250,
                            )

                    with st.expander("🔎 Trace truy xuất", expanded=False):
                        dq_trace = response.get("dq_context_trace", [])
                        gq_trace = response.get("gq_context_trace", [])
                        if not dq_trace and not gq_trace:
                            st.info("Không có trace truy xuất.")
                        if dq_trace:
                            st.caption("Dq trace")
                            st.json(dq_trace)
                        if gq_trace:
                            st.caption("Gq trace")
                            st.json(gq_trace)

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

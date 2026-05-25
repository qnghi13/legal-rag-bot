import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.bot import get_hr_bot

st.set_page_config(page_title="Labor Law Bot", page_icon="🤖")
st.title("Chat bot luật Lao động Việt Nam")
st.markdown("Xin chào! Tôi có thể giúp gì cho bạn?")

@st.cache_resource
def load_bot():
    return get_hr_bot()

rag_chain = load_bot()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Lưu lịch sử chat dạng object của LangChain để nạp vào mô hình
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
                # Gọi Chain RAG LCEL
                response = rag_chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history 
                })
                
                # Bóc tách Dữ liệu
                bot_reply = response["answer"]
                context_used = response["context"] # <--- Lấy ngữ cảnh ra
                
                # In câu trả lời
                st.markdown(bot_reply)
                
                # HIỂN THỊ LẠI Ô XEM TÀI LIỆU Ở ĐÂY
                with st.expander("📄 Xem tài liệu trích xuất"):
                    if context_used.strip() == "":
                        st.warning("Không tìm thấy ngữ cảnh nào phù hợp trong PDF.")
                    else:
                        st.text_area("Văn bản AI đã đọc:", value=context_used, height=250)
                
                # Cập nhật lịch sử
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                st.session_state.chat_history.extend([
                    HumanMessage(content=prompt),
                    AIMessage(content=bot_reply)
                ])

            except Exception as e:
                st.error(f"Có lỗi xảy ra: {e}")
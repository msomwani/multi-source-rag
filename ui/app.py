import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Multi-Source RAG",
    layout="wide"
)

st.title("🧠 Multi-Source RAG System")
st.write("Upload documents or ingest websites, then ask questions.")

st.sidebar.header("📥 Ingest Data")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])


uploaded_file = st.sidebar.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True
)

if uploaded_file:
    if st.sidebar.button("Ingest Files"):
        with st.spinner("Ingesting files..."):
            success = 0
            for file in uploaded_file:
                files = {"file": file}
                response = requests.post(
                    f"{BACKEND_URL}/ingest/file",
                    files=files
                )
                if response.status_code == 200:
                    success += 1

        st.sidebar.success(f"Ingested {success} files successfully!")



st.sidebar.divider()
st.sidebar.subheader("🌐 Ingest Website")

url = st.sidebar.text_input("Website URL")

if url:
    if st.sidebar.button("Ingest URL"):
        with st.spinner("Fetching and ingesting website..."):
            response = requests.post(
                f"{BACKEND_URL}/ingest/url",
                data={"url": url}
            )

        if response.status_code == 200:
            st.sidebar.success("Website ingested successfully!")
            st.sidebar.json(response.json())
        else:
            st.sidebar.error(response.text)


st.divider()
st.header(" Ask a Question ❓")

# question = st.text_input("Enter your question")

# if question:
#     if st.button("Ask"):
#         with st.spinner("Thinking..."):
#             response = requests.post(
#                 f"{BACKEND_URL}/query",
#                 data={"question": question}
#             )

#         if response.status_code == 200:
#             result = response.json()

#             st.subheader("✅ Answer")
#             st.write(result["answer"])

#             st.subheader("📚 Sources")
#             for src in result["sources"]:
#                 st.json(src)
#         else:
#             st.error(response.text)


question = st.chat_input("Ask a question about your documents")
if question:
    # 1. Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    # 2. Call backend
    with st.spinner("Thinking..."):
        response = requests.post(
            "http://localhost:8000/query",
            json={"question": question}
        )

    if response.status_code != 200:
        answer_text = "Error contacting backend."
        sources = []
    else:
        data = response.json()
        answer_text = data.get("answer", "")
        sources = data.get("sources", [])

    # 3. Store assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources
    })

    # 4. Render assistant reply
    with st.chat_message("assistant"):
        st.write(answer_text)
        if sources:
            with st.expander("Sources"):
                for src in sources:
                    st.write(src)

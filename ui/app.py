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

question = st.text_input("Enter your question")

if question:
    if st.button("Ask"):
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{BACKEND_URL}/query",
                data={"question": question}
            )

        if response.status_code == 200:
            result = response.json()

            st.subheader("✅ Answer")
            st.write(result["answer"])

            st.subheader("📚 Sources")
            for src in result["sources"]:
                st.json(src)
        else:
            st.error(response.text)

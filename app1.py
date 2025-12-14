import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch

# ---- LangChain 2025 imports ----
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# --------------------------------

# ============================================================
# 1. CONFIG (YOU control retrieval depth here)
# ============================================================
RETRIEVAL_TOP = 3   # 🔒 AzureSearch uses `top`, NOT `k`

# ============================================================
# 2. Load Secrets
# ============================================================
load_dotenv()

# ============================================================
# 3. Streamlit Page Setup
# ============================================================
st.set_page_config(page_title="Azure RAG Bot", page_icon="🤖")
st.title("🤖 Chat with your PDF")

# ============================================================
# 4. Build RAG Chain (cached)
# ============================================================
@st.cache_resource
def get_rag_chain(top: int):
    # ---- Embeddings ----
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
    )

    # ---- Azure AI Search Vector Store ----
    vector_store = AzureSearch(
        azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
        embedding_function=embeddings.embed_query,
    )

    # ---- LLM ----
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        temperature=0,
    )

    # ---- Prompt ----
    system_prompt = (
        "You are a helpful assistant. "
        "Use the following context to answer the user's question. "
        "If you don't know the answer, say you don't know.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # ---- QA Chain ----
    qa_chain = create_stuff_documents_chain(llm, prompt)

    # ✅ FINAL FIX: AzureSearch-safe retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"top": top},
    )

    # ---- RAG Chain ----
    return create_retrieval_chain(retriever, qa_chain)

# ============================================================
# 5. Initialize Chain
# ============================================================
try:
    chain = get_rag_chain(RETRIEVAL_TOP)
except Exception as e:
    st.error(f"Error connecting to Azure services: {e}")
    st.stop()

# ============================================================
# 6. Chat History
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============================================================
# 7. Chat Loop
# ============================================================
if user_input := st.chat_input("Ask a question about your PDF..."):
    # ---- User message ----
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    # ---- Assistant message ----
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chain.invoke({"input": user_input})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as e:
                st.error(f"Error generating response: {e}")

import streamlit as st
import os
import re
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.messages import SystemMessage, HumanMessage

# ============================================================
# 1. CONFIG
# ============================================================
RETRIEVAL_TOP = 12  # Azure uses `top`, not `k`

# ============================================================
# 2. Load Secrets
# ============================================================
load_dotenv()

# ============================================================
# 3. Streamlit Setup
# ============================================================
st.set_page_config(page_title="Azure Hybrid RAG Bot", page_icon="🤖")
st.title("🤖 Chat with your PDF")

# ============================================================
# 4. Build Components (cached)
# ============================================================
@st.cache_resource
def build_components(top: int):

    # ---- Embeddings ----
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
    )

    # ---- Azure AI Search ----
    vector_store = AzureSearch(
        azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
        embedding_function=embeddings.embed_query,
    )

    retriever = vector_store.as_retriever(
        search_type="hybrid",
        k=RETRIEVAL_TOP,
    )

    # ---- LLM ----
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        api_version=os.getenv(" "),
        temperature=0,
    )

    return retriever, llm


# ============================================================
# 5. Initialize
# ============================================================
try:
    retriever, llm = build_components(RETRIEVAL_TOP)
except Exception as e:
    st.error(f"Startup error: {e}")
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
# 7. Chat Loop (FINAL + FIXED)
# ============================================================
if user_input := st.chat_input("Ask a question about your PDF..."):

    # ---- User ----
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    # ---- Assistant ----
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # ------------------------------------------------
                # 1️⃣ Retrieve documents (HYBRID)
                # ------------------------------------------------
                docs = retriever.invoke(user_input)

                #print block to see retrieved docs

                print(f"\n--- DEBUG: Retrieved {len(docs)} chunks ---")
                
                # Create a collapsible section in Streamlit so it doesn't clutter the UI
                with st.expander(f"🔍 Debug: View all {len(docs)} retrieved chunks"):
                    for i, doc in enumerate(docs):
                        source_page = doc.metadata.get("page", "N/A")
                        preview = doc.page_content.replace("\n", " ")[:100] # Clean preview for terminal

                        # # 1. Print to Terminal
                        # print(f"[Chunk {i}] [Page {source_page}] {preview}...")

                        # 2. Show in Streamlit
                        st.markdown(f"**Chunk {i} | Page {source_page}**")
                        st.text(doc.page_content) # st.text preserves formatting better for raw text
                        st.divider()

                if not docs:
                    st.markdown("I don't know.")
                    st.stop()

                # ------------------------------------------------
                # 2️⃣ Build STRING context with CHUNK IDs
                # ------------------------------------------------
                context_blocks = []
                for i, d in enumerate(docs):
                    page = d.metadata.get("page", "N/A")
                    context_blocks.append(
                        f"[CHUNK {i} | PAGE {page}]\n{d.page_content}"
                    )

                context = "\n\n".join(context_blocks)

                # ------------------------------------------------
                # 3️⃣ System Prompt (LLM must choose chunk)
                # ------------------------------------------------
                system_prompt = f"""
You are a precise question-answering assistant.

You will be given document chunks.
Each chunk has a CHUNK ID and PAGE number.

Rules:
- Use ONLY the given chunks
- Answer the question
- Choose EXACTLY ONE chunk that supports the answer
- You may make simple factual inferences
- If no chunk answers the question, say "I don't know"

Respond in EXACTLY this format:

ANSWER: <answer>
CHUNK_ID: <number>
PAGE: <number>

Context:
{context}
"""

                # ------------------------------------------------
                # 4️⃣ Call LLM DIRECTLY (NO chain)
                # ------------------------------------------------
                response = llm.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_input),
                    ]
                )

                raw_text = response.content
                # print(raw_text)

                # ------------------------------------------------
                # 5️⃣ Parse LLM output
                # ------------------------------------------------
                answer_match = re.search(r"ANSWER:\s*(.*)", raw_text)
                chunk_match = re.search(r"CHUNK_ID:\s*(\d+)", raw_text)

                if not answer_match or not chunk_match:
                    st.markdown("I don't know.")
                    st.stop()

                answer = answer_match.group(1).strip()
                chunk_id = int(chunk_match.group(1))

                if chunk_id < 0 or chunk_id >= len(docs):
                    st.markdown("I don't know.")
                    st.stop()

                # ------------------------------------------------
                # 6️⃣ Display Answer
                # ------------------------------------------------
                st.markdown(answer)

                # ------------------------------------------------
                # 7️⃣ Display EXACT supporting chunk
                # ------------------------------------------------
                supporting_doc = docs[chunk_id]

                page_number = supporting_doc.metadata.get("page", "N/A")
                source_name = supporting_doc.metadata.get("source", "Document")

                st.markdown("---")
                st.markdown("### 📄 Supporting Evidence")
                st.markdown(f"**Source:** {source_name}")
                st.markdown(f"**Page:** {page_number}")
                st.markdown("**Excerpt:**")
                st.markdown(supporting_doc.page_content)

                # ------------------------------------------------
                # 8️⃣ Save history
                # ------------------------------------------------
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                st.error(f"Error generating response: {e}")
import json
import urllib.request

import streamlit as st

from dense_retrieval import DenseRetriever


MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"
TOP_K = 5


@st.cache_resource
def get_retriever():
    return DenseRetriever()


def ask_llama(question, passages):

    context = "\n\n".join(
        f"[Source {i+1}]\n{p.get('text', '')}"
        for i, p in enumerate(passages)
    )

    prompt = f"""You are a question-answering system for a medical and nutrition information database.

Answer the question using ONLY the information in the provided context.

Do not use outside knowledge.
Do not invent facts.

If the context does not contain enough information, say:
"Insufficient information in the retrieved context."

Give a concise factual answer.

Question:
{question}

Context:
{context}

Answer:
"""

    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(
            response.read().decode("utf-8")
        )

    return result["response"].strip()


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------

st.set_page_config(
    page_title="NFCorpus Domain QA",
    page_icon="🔎",
    layout="wide"
)

st.title("NFCorpus Domain-Specific Question Answering")

st.caption(
    "BGE dense retrieval + Llama 3.1 8B | "
    "Local inference | No API key required"
)

question = st.text_input(
    "Ask a question",
    placeholder="e.g. Does vitamin D reduce the risk of cancer?"
)

ask_clicked = st.button(
    "Ask",
    type="primary"
)

if ask_clicked:

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Retrieving relevant documents..."):

            retriever = get_retriever()

            passages = retriever.search(
                question.strip(),
                k=TOP_K
            )

        with st.spinner("Generating answer with Llama 3.1..."):

            try:

                answer = ask_llama(
                    question.strip(),
                    passages
                )

            except Exception as e:

                st.error(
                    "Could not connect to Ollama. "
                    "Make sure Ollama is running."
                )

                st.exception(e)
                st.stop()

        st.subheader("Answer")

        st.write(answer)

        st.subheader("Retrieved Evidence")

        for i, passage in enumerate(passages, 1):

            doc_id = passage.get(
                "doc_id",
                "unknown"
            )

            with st.expander(
                f"Source {i} — {doc_id}"
            ):

                st.write(
                    passage.get(
                        "text",
                        ""
                    )
                )
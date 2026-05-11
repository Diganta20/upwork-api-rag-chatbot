import streamlit as st
import time

from rag_pipeline import FALLBACK_ANSWER, answer_question


st.set_page_config(
    page_title="Upwork API RAG Chatbot",
    page_icon=":speech_balloon:"
)

st.markdown(
    """
    <h1 style='text-align: center;'>
        Upwork API RAG Chatbot
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='text-align: center; margin-top: -15px; color: gray;'>
        Answers are generated only from retrieved
        Upwork API documentation chunks.
    </p>
    """,
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about the Upwork API documentation")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documentation..."):

            start_time = time.time()

            result = answer_question(question)

            end_time = time.time()
            latency = round(end_time - start_time, 2)

        answer = (
            result["answer"].strip()
            or FALLBACK_ANSWER
        )

        st.markdown(answer)

        st.caption(
            f"⏱ Response time: {latency} seconds"
        )

        with st.expander("Source chunks used"):
            if result["sources"]:
                for index, source in enumerate(
                    result["sources"],
                    start=1
                ):
                    st.markdown(
                        f"**Source {index}: "
                        f"`{source['source']}`**"
                    )
                    st.write(source["content"])
            else:
                st.write(
                    "No source chunks were retrieved."
                )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
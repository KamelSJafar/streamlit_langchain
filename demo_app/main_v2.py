"""Python file to serve as the frontend"""
import sys
import os
import time
sys.path.append(os.path.abspath('.'))

import streamlit as st
from demo_app.components.sidebar import sidebar

def ingest_data(data_urls):
    print({data_urls})
    naval_chat_bot.add("web_page", data_urls['web_page_1'])
    st.session_state["IS_BOT_READY"] = True

def response_embedchain(query):
    """Logic for loading the chain you want to use should go here."""
    try:
        #print(f'Calling response on: {query}')
        sys_prompt = """
        Act as Sentiment analysis system and I will give you a context from the Article you will give me the response in JSON format for example:
        {"topic":"sport or media or political or etc" ,"related_to_UAE": "true or false", 
        "sentiment":{"positive": {"score": 60%, "example":"positive sentence from the article"}, 
        "negative":{"score": 30%, "example":"negative sentence from the article"}, 
        "neutral":{"score": 10%, "example":"neutral sentence from the article"}}, 
        "NER":"all NER mentioned in the article"}
        """
        response = naval_chat_bot.query(sys_prompt)
        #print(response)
        return response
    except Exception as e:
        print(e, "###########################")

def provide_data_urls():
    with st.expander("Source Data Form", expanded=st.session_state["expander_state"]):
        form = st.form(key="source-data", clear_on_submit=False)
        web_page_1 = form.text_input("Enter URL: web page",
                                     autocomplete="https://nav.al/feedback",
                                     placeholder="https://nav.al/feedback")
        submit_data_form = form.form_submit_button("Submit", on_click=toggle_closed)

        if submit_data_form:
            st.session_state["submit_data_form"] = True

    data_dict = {'web_page_1': web_page_1}
    return data_dict

def toggle_closed():
    st.session_state["expander_state"] = False

if __name__ == "__main__":
    st.set_page_config(
        page_title="Chat App: EmbedChain Demo",
        page_icon="📖",
        layout="wide",
        initial_sidebar_state="expanded", 
    )
    st.header("📖 Chat App: EmbedChain Demo")

    sidebar()

    if "expander_state" not in st.session_state:
        st.session_state["expander_state"] = True

    data_dict = provide_data_urls()

    if not st.session_state.get("OPENAI_API_CONFIGURED") or not st.session_state.get("submit_data_form"):
        st.error("Please configure your API Keys! and Submit the form")

    if st.session_state.get("OPENAI_API_CONFIGURED") and st.session_state.get("submit_data_form"):
        st.markdown("Main App: Started")
        from embedchain import App as ecApp

        naval_chat_bot = ecApp()
        # ingesting data
        try:
            if not st.session_state.get("IS_BOT_READY"):
                with st.spinner('Wait for DATA Ingestion'):
                    ingest_data(data_dict)
                st.success('Data Ingestion Done!')

            if st.session_state.get("IS_BOT_READY"):
                if "messages" not in st.session_state:
                    st.session_state["messages"] = [
                        {"role": "assistant", "content": "How can I help you?"}]

                # Directly trigger the response on form submission
                user_input = "Provide a prompt here if needed"  # You can set a default prompt or leave it dynamic
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    with st.spinner('CHAT-BOT is at Work ...'):
                        assistant_response = response_embedchain(user_input)
                    #print(assistant_response, "assistant_response")
                    try:
                        for chunk in assistant_response.split():
                            full_response += chunk + " "
                            time.sleep(0.05)
                            message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)
                    except Exception as e:
                        print(e, "###########################")
                st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            print(e, "###########################")

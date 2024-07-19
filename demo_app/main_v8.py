import re
import json
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader


def setup_openai_client():
    """Setup the OpenAI client."""
    client = openai.OpenAI(
        api_key="ZadDDgicE6mcXS2sxt8j2ZlPi0A0gIFi",
        base_url="https://api.deepinfra.com/v1/openai"
    )
    return client
def load_document(url):
    """Load document from the specified URL."""
    loader = WebBaseLoader(url)
    return loader.load()
def extract_summarization_response(text):
    pattern = r'<RES>(.*?)</RES>'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    json_str = match.group(1)
    try:
        result = json.loads(json_str)
        return result
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        return None


def summarize_document(client, text):

    """Summarize the document using OpenAI client."""
    try:
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-70B-Instruct",
            messages=[
                {"role": "system", "content": """Act as Sentiment analysis system and I will give you a context from the Article you will give me the response in JSON format for example:
        {"topic":"sport or media or political or etc" ,"related_to_UAE": "true or false", 
        "summary":"short summary about the Article"
        "sentiment":{"positive": {"score": 60, "example":"positive sentence from the article"}, 
        "negative":{"score": 30, "example":"negative sentence from the article"}, 
        "neutral":{"score": 10, "example":"neutral sentence from the article"}}, 
        "NER":"all NER mentioned in the article"}}
                    Wrap the JSON Response inside <RES> tags
                    """},
                {"role": "user", "content": text}
            ],
            
        )
        print(response.choices[0].message.content)
        friars = extract_summarization_response(response.choices[0].message.content)
        print(friars)
        

        return friars
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def main():
    load_dotenv()
    st.set_page_config(page_title="Summarize Web Content")
    st.header("Summarize Web Content 🌐")
    
    web_page_1 = st.text_input("Enter URL for web page",
                               autocomplete="https://example.com",
                               placeholder="https://example.com")

    if web_page_1:
        urls = [web_page_1]
        docs = load_document(urls)

        # print (docs[0].page_content)
        # Summarize the document using OpenAI client
        client = setup_openai_client()
        summary = summarize_document(client, docs[0].page_content)
        print (summary)
        print ("***********************************************************************")

        if summary:
            st.write("Summarized document:")
            st.json(summary)
        else:
            st.write("No content found at the provided URL.")
    else:
        st.write("Please enter a valid web page URL.")

if __name__ == '__main__':
    main()

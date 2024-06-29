from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import AsyncHtmlLoader

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your Document")
    st.header("Ask your Document 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    web_page_1 = st.text_input("Enter URL for web page",
                               autocomplete="https://nav.al/feedback",
                               placeholder="https://nav.al/feedback")

    text = ""

    # extract text from PDF
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    # extract text from web page
    if web_page_1:
        urls = [web_page_1]
        loader = AsyncHtmlLoader(urls)
        web_docs = loader.load()
        for doc in web_docs:
            text += doc.page_content

    # ensure text is not empty before processing
    if text:
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # create embeddings
        ollama_emb = OllamaEmbeddings(
            model="nomic-embed-text",
        )

        knowledge_base = FAISS.from_texts(chunks, ollama_emb)
        
        # show user input
        user_question = st.text_input("Ask a question about your document:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = ChatOllama(model="llama3")
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.invoke({
                    "input_documents": docs,
                    "question": user_question
                })
                print(cb)
                
            st.write(response)
    else:
        st.write("Please upload a PDF or enter a valid web page URL.")

if __name__ == '__main__':
    main()

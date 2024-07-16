import argparse

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader


def setup_argparse():
    """Setup argparse to parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize a document from a given URL."
    )
    parser.add_argument(
        "-u", "--url", required=True, help="URL of the document to summarize"
    )
    return parser.parse_args()


def load_document(url):
    """Load document from the specified URL."""
    loader = WebBaseLoader(url)
    return loader.load()


def setup_summarization_chain():
    """Setup the summarization chain with a prompt template and ChatOllama."""
    prompt_template = PromptTemplate(
        template="""As a professional summarizer, create a detailed and comprehensive summary in Arabic language of the provided text, be it an article, post, conversation, or passage, while adhering to these guidelines:
            
            1. Craft a summary that is detailed, thorough, in-depth, and complex, while maintaining clarity.

            2. Incorporate main ideas and essential information, eliminating extraneous language and focusing on critical aspects.

            3. Rely strictly on the provided text, without including external information.

            4.Extract the following information from the provided text:

                # PAPERSNAPv1: Exact information of Appeal
                ## 🎓 معلومات :
                -هيئة المحكمة: Name all memeber in sequence.
                
                - Appeal No: Exact number of the Appeal.
                
                - type: Specific the type of Appeal like إداري، تجاري، تأديب محامين,.....
                
                - Publication Year: Year the Appeal.
                
                - DOI/URL: Direct link or unique identifier.

            5. Format the summary in multi-paragraph form for easy understanding include الملخص ، النقاط الرئيسية ، النتيجة ،....

            6. Make sure that the summary is exclusively in classical Arabic.

            7. Be careful that the summary don't include any character not Arabic.

            8. If the summary not in Arabic language work as a professional translator,and provide a detailed and comprehensive translation of the resulted  summary into Arabic,ensuring that the translation is accurate and coherent of the result.
                             
            9.Conclude your notes with [End of Notes, Message #X] to indicate completion, where "X" represents the total number of messages that I have sent. In other words, include a message counter where you start with #1 and add 1 to the message counter every time I send a message.
           
            

        By following this optimized prompt, you will generate an effective summary that encapsulates the essence of the given text in a clear, detailed, and reader-friendly manner. Optimize output as markdown file.

        "{text}" 
    
        DETAILED SUMMARY IN ARABIC LANGUAGE: 
        ## 🎓 الملخص :
        ## 🎓النقاط الرئيسية:
        ## 🎓النتيجة: """,
        input_variables=["text"],
    )

    llm = ChatOllama(model="llama3:latest", base_url="http://127.0.0.1:11434")
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    return llm_chain



def main():
    args = setup_argparse()
    docs = load_document(args.url)

    llm_chain = setup_summarization_chain()
    result = llm_chain.run(docs)


if __name__ == "__main__":
    main()

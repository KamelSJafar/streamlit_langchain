import re
import json
import openai
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader

app = Flask(__name__)

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
                {"role": "system", "content": """Act as Data Analysis System,Your answer must be in the same language as the article. I will give you an Article you will give me the response in JSON format for example:
        {"topic":"sport or media or political or etc" ,"related_to_UAE": "true or false",
        "related_to_UAE_percentage": "Percentage of the article being related to the United Arab Emirates the language of the answer MUST be as the Article language", 
        "summary":"short summary about the Article"
        "sentiment":{"positive": {"score": 60, "example":"positive sentence from the article"}, 
        "negative":{"score": 30, "example":"negative sentence from the article"}, 
        "neutral":{"score": 10, "example":"neutral sentence from the article"}}, 
        "NER":"all NER mentioned in the article"}}
                    Wrap the JSON Response inside <RES> tags
                 Your Answer Must be in the same language of the Article.
                    """},
                {"role": "user", "content": text}
            ],
        )
        print(response.choices[0].message.content)
        friars = extract_summarization_response(response.choices[0].message.content)
        print(friars)

        return friars
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    docs = load_document(url)
    if not docs:
        return jsonify({"error": "Failed to load document"}), 500
    
    client = setup_openai_client()
    summary = summarize_document(client, docs[0].page_content)
    if summary:
        return jsonify(summary), 200
    else:
        return jsonify({"error": "Failed to summarize document"}), 500

if __name__ == '__main__':
    load_dotenv()
    app.run(host='0.0.0.0', port=5000)

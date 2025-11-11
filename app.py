from flask import Flask, render_template,jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
import pinecone
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from dotenv import load_dotenv
import os
from src.prompt import *
from pinecone import Pinecone as PineconeClient, ServerlessSpec
app = Flask(__name__)
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
embeddings = download_hugging_face_embeddings()

pc=PineconeClient(api_key=PINECONE_API_KEY)

docsearch = LangchainPinecone.from_existing_index(index_name="medicalbot",
    embedding=embeddings)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

# Load the model through LangChain wrapper
llm = CTransformers(
    model=r"C:\Users\mouny\medical-chatbot\model\llama-2-7b-chat.Q4_0.gguf",
    model_type="llama",
    config={
        'gpu_layers': 0,
        'max_new_tokens': 512,
        'temperature': 0.7,
        'context_length': 2048
    }
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def get_response():
    user_msg = request.json.get("message", "")
    if user_msg in ["hello", "hi", "hey"]:
        return jsonify({"answer": "Hello, how can I help you?"})

    
    result = qa.invoke({"query": user_msg})
    return jsonify({"answer": result["result"]})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8020, debug=True)

   
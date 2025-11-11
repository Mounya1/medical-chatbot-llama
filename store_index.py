from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
import pinecone
from dotenv import load_dotenv
import os

from pinecone import Pinecone as PineconeClient, ServerlessSpec
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

#print(PINECONE_API_KEY)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
pc=PineconeClient(api_key=PINECONE_API_KEY)
index_name = "medicalbot"




#Creating Embeddings for Each of The Text Chunks & storing
docsearch=LangchainPinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)

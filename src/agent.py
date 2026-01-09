import os
import bs4
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

#get the Groq api key 
groq_api_key = os.getenv("GROQ_API_KEY")

#Call. the llm model 
llm = ChatGroq(groq_api_key = groq_api_key, model_name = "llama-3.1-8b-instant")

#initialize the Hugging face embeddings
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
#define the webbase loader


loader = WebBaseLoader(
    web_paths = ("https://mlu-explain.github.io/","https://mlu-explain.github.io/logistic-regression/"),
    # bs_kwargs = dict(
    #     # parse_only = bs4.SoupStrainer(
    #     #     class_ = ("post-content","post-title","post-header")
    #     # )
    # ),
)
#load the page into documents
docs = loader.load() 
# print(len(docs))


#Split the documents 
#initialize the the splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
splits = text_splitter.split_documents(docs)

# print(len(splits))

# store them in a chrooma db
Vector_store = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)

"""Note :- we have to use same embedding for while retrival because if we use different the similarity search may not work"""

#define the retriever

retriever =  Vector_store.as_retriever()

# print(f"{retriever}")


#Define the Prompt Templete

system_prompt = ("""
You are a professional Machine learning tutor
Answer the question related to Machine Learning With precise pieces of
retrieved context to the answer.If You don't know the answer please don't answer 
Understand the problem statement before answering. and explain the sloution clearly to user.B
But never answer the question that is not related to retrieved context
"""
                 "\n\n"
                 "{context}")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}"),
    ]
)


# Helper to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


question_answer_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough(),
    }
    | prompt
    | llm
)

# Run
user = input("enter the query: ")
response = question_answer_chain.invoke(user)

print(response.content)


##Now We are adding the chat History

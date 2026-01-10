import os
import bs4
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

# response = question_answer_chain.invoke(user)

# print(response.content)

##Now We are adding the chat History
context_system_prompt = """

You are a query rewriter for a retrieval system.
ONLY rewrite the user's question if it clearly depends on previous chat history
uses words like "it", "that", "this", "they", "explain more", "what about".
If the question is already complete and standalone,
return it EXACTLY as written.
Do NOT mention chat history.
Do NOT explain anything.
Do NOT answer the question.
Only output the rewritten or original question.
"""

context_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", context_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
contextualize_question = context_prompt | llm

history_aware_retriever = (
    contextualize_question
    | RunnableLambda(lambda msg: msg.content)
    | retriever
)

#now we have toUpdate our chain to use the history of context
question_answer_chain = (
    {
        "context": history_aware_retriever | format_docs,
        "input": RunnablePassthrough(),
    }
    | prompt
    | llm
)
chat_history = []

print("\nType 'exit' or 'quit' to end the chat.\n")

while True:
    user = input("You: ").strip()
    if user.lower() in {"exit", "quit"}:
        print("Goodbye")
        break

    response = question_answer_chain.invoke(
        {
            "input": user,
            "chat_history": chat_history,
        }
    )

    print("\nAssistant:\n", response.content, "\n")

    # update chat history
    chat_history.append(("human", user))
    chat_history.append(("ai", response.content))

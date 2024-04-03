import streamlit as st
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


@st.cache_resource(show_spinner=False) #Only runs once
def get_retriever():
    # Loading the saved embeddings 
    embedding_model = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    loaded_vectors=FAISS.load_local("vectors", embedding_model)

    faiss_retriever = loaded_vectors.as_retriever(search_kwargs={"k":2})

    return faiss_retriever

def get_chain():
    PROMPT_TEMPLATE = '''
    You are a helpful military chatbot.
    With the information being provided try to answer the user question. 
    If you cant answer the question based on the information just say that you are unable to find an answer.
    Try to deeply understand the context and answer only based on the information provided. 
    Dont generate irrelevant answers. Do provide only helpful answers.

    Context: {context}
    Question: {question}
    
    Helpful answer:
    '''
    #Context are the chunks retrived from the database, the Question is the user prompt
    input_variables = ['context', 'question']
   
    custom_prompt = PromptTemplate(template=PROMPT_TEMPLATE,
                                input_variables=input_variables)

    with st.spinner("Loading..."):
        faiss_retriever = get_retriever()

    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.5)

    #Finds the relevant chunks using the faiss_retriver and returns the LLM response
    qa_with_sources_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever = faiss_retriever,
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True
    )
    return qa_with_sources_chain

#To create a query with contextual knowledge
def generate_refined_query(new_query, chat_history):
    conversation_so_far = ""
    for query, response in chat_history:
        conversation_so_far += "\nUser: " + query + "\nAI: " + response
    
    prompt = """
                I have built a chatbot and I need help with rewriting the user query.
                First assess whether the "New User Query" is complete and makes sense independent of the chat history.
                If it does, simply return the query as-is without making any change to it.
                If it doesn't, rewrite the new user query by filling in the missing information from chat history 
                such that the re-written query is complete and makes sense independent of the chat history.
                At any point of time, do not add any information from your own side. 
                I will give you 1 example of each scenario to explain the requirements.

                Example 1: When new user query is dependent on chat history and requires rewriting
                Chat Conversation History:
                User: Who is the prime minister of India?
                AI: Narendar Modi
                
                New User Query: How old is he?

                Response: How old is Narendar Modi?

                Example 2: When new user query is complete and independent and doesn't require rewriting
                Chat Conversation History:
                User: Who is the prime minister of India?
                AI: Narendar Modi
                
                New User Query: Who is the prime minister of England?

                Response: Who is the prime minister of England?

                Real: This is not an example. You need to respond to this one.
                Chat Conversation History: """ + conversation_so_far + """\n\nNew User Query: """ + new_query + "\n\n Response:"
    
    print(prompt)
    chat = ChatGoogleGenerativeAI(model="gemini-pro")
    return chat.invoke(prompt).content

def query_index(qa_with_sources_chain, query, chat_history):
    if len(chat_history) == 0:
        refined_query = query
    else:
        refined_query = generate_refined_query(query, chat_history)
    print(refined_query)

    response = qa_with_sources_chain({"query":refined_query})
    print(response["result"])
    return response

st.header("Military Chatbot 🪖")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"]=[]
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"]=[]
if "chat_history" not in st.session_state:
    st.session_state["chat_history"]=[]

hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

prompt = st.chat_input("Enter your question here")
qa_with_sources_chain = get_chain()

if prompt:
    with st.spinner("Generating..."):
        output=query_index(qa_with_sources_chain, query=prompt, chat_history = st.session_state["chat_history"])
        st.session_state["chat_answers_history"].append(output['result'])
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_history"].append((prompt,output['result']))

if st.session_state["chat_answers_history"]:
    for i, j in zip(st.session_state["chat_answers_history"],st.session_state["user_prompt_history"]):
        message1 = st.chat_message("user")
        message1.write(j)
        message2 = st.chat_message("assistant")
        message2.write(i)
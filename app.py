from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
# from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.callbacks import get_openai_callback
import os
import torch
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm

# Read the pdffiles
def get_pdf_text(pdf_docs):
    text =""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        # pdf_reader = PDFPlumberLoader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Splitting up the text into smaller chunks for indexing
def get_text_chunk(text):
    text_splitter = CharacterTextSplitter(separator="\n",chunk_size=1000, chunk_overlap=200, length_function=len) # character size is charaters
    chunks = text_splitter.split_text(text=text)
    #print(chunks)
    return chunks
# creating embeddings
def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()  #have charge to pay
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")  # trying for free
    #embeddings = GooglePalmEmbeddings() ## google 
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings) # knowledge base from our pdfile
    return vectorstore

def get_conversation_chain(vectorstore):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #llm = ChatOpenAI()  #
    #llm = GooglePalm()
    llm = HuggingFaceHub(repo_id= "google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512, "device": device})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True) #storing every chat interaction directly in the buffer.
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever= vectorstore.as_retriever(),
        memory=memory

    )
    # print(memory.buffer)
    return conversation_chain

def handle_userinput(user_questions):
    response = st.session_state.conversation({'question': user_questions})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        # st.write(i,message)
        if i%2 == 0:
            st.write("Questions: ", message.content)
        #     # st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write("Bot Response: ", message.content)
        #     # st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
# streamlit app code
def main():
    load_dotenv()
    # print(os.getenv("OPENAI_API_KEY"))  # check whether it's track of ke.

    # print("HI")
    #let’s set the page configuration and have a session state of the class to avoid instantiating the class multiple times in the same session.
    st.set_page_config(page_title="Chat with multiple pdfs", page_icon=":books:",layout='wide',initial_sidebar_state='auto')
    # initialise conversaion and chat_history within streamlit session state
    if "conversation" not in st.session_state:  # if session is not open 
        st.session_state.conversation = None
        # st.session_state['conversation'] = ["How can I assist you?"]

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        # st.session_state['chat_history'] = []
    st.header("Chat with Mulitple pdfs :books:")
    #st.text_input("Ask a question about your documents:")
    user_questions = st.text_input("Ask a question about your documents:")
    if user_questions:
        # pass
        handle_userinput(user_questions)
    
    # st.write(user_template.replace("{{MSG}}", ""))
    with st.sidebar:
        st.subheader("Read your documents")
        pdf_docs = st.file_uploader(label="upload your pdf here and click on processs",accept_multiple_files=True )
        if st.button("process"):
            with st.spinner("Processing"):
                # get the text
                raw_text = get_pdf_text(pdf_docs)


                # get the text chunks
                text_chunks = get_text_chunk(raw_text)


                # create vector store
                vectorstore =  get_vectorstore(text_chunks)
                
                ## it charge for use of openai
    #             if user_questions:
    #                 docs = vectorstore.similarity_search(user_questions)

    # #             ## create conversation chain
    #                 llm= OpenAI()
                    # llm = OpenAI(model_name="text-davinci-003", n=2, best_of=2)
                    ## Load a QA chain using an OpenAI object, a chain type, and a prompt template.
    #                 chain = load_qa_chain(llm, chain_type='stuff') # # we are going to stuff all the docs in at once
                    # with get_openai_callback() as pricecheck:         
                    #     response = chain.run(input_documents=docs, question=user_questions) # result from knowledge base and passed to llm
                    #     print(pricecheck)
    #                 st.write(response)

                # create conservation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Done")

if __name__== '__main__':
    main()
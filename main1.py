__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# from dotenv import load_dotenv
# load_dotenv()

from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os

# 제목
st.title("ChatPDF")
st.write("---")

# 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요.", type = ['pdf'])
st.write("---")

def pdf_to_document(uploaded_file) :
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f :
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# 업로드 후 동작
if uploaded_file is not None :
    pages = pdf_to_document(uploaded_file)
   
    # split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=300,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.split_documents(pages)

    # Embedding
    embeddings_model = OpenAIEmbeddings()

    # load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)

     # Question
    st.header("PDF에게 질문해 보세요!!")
    question = st.text_input("질문을 입력하세요.")

    if st.button("질문하기") :
        with st.spinner('실행 중...') :
            llm_u = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm=llm_u, retriever=db.as_retriever())
            result = qa_chain.invoke(dict(query=question))
            st.write(result["result"])

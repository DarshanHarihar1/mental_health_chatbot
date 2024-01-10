from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


import warnings
from warnings import filterwarnings
filterwarnings('ignore')

llm = GooglePalm(google_api_key = 'AIzaSyAiM4eafhSL66-R3HNh3wrSjZnTJqZTBk0', temperature = 0.1)

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name = 'hkunlp/instructor-large')
vectordb_file_path = 'faiss_index'

def create_vector_db():
    loader = CSVLoader(file_path = 'samplee.csv', source_column = 'prompt', encoding='utf-8')
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)
    
    
def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)
    retriever = vectordb.as_retriever(score_threshold = 0.7)
    prompt_template = """Given the following context and a question, generate an answer based on this context only. In the answer, try to provide as much text as possible from "response" section in the source document context without making many changes. If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.
    CONTEXT: {context}
    QUESTION: {question}"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain_type_kwargs = {"prompt": PROMPT}
    chain = RetrievalQA.from_chain_type(llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            input_key="query",
                            return_source_documents=True,
                            chain_type_kwargs=chain_type_kwargs)
    
    return chain
    

if __name__ == '__main__':
    chain = get_qa_chain()
    
    print(chain('who is elon musk'))


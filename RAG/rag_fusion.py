import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import loads, dumps
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.vectorstores import FAISS   
from pydantic import BaseModel, AfterValidator, Field, ValidationError
import instructor
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
client = instructor.from_gemini(
client=genai.GenerativeModel(
    model_name="models/gemini-1.5-flash-latest",  # model defaults to "gemini-pro"
),
mode=instructor.Mode.GEMINI_JSON,
)

class LLMResponse(BaseModel):
    question: str
    isValidResponse: bool = Field(description="True if the answer is related to Kubernetes, AWS, linux, docker, networking or it's related topics, False otherwise")
    

def initGeminiLLM():
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5, max_retries=3)
    return llm

def loadVectorDB(folder_path):
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vstore2 = FAISS.load_local(folder_path=folder_path, index_name="blog", embeddings=embedding_function, allow_dangerous_deserialization=True)
    return vstore2

def filterQueries(queries):
    filtered_queries = [d.strip() for d in queries if d.strip()!='']
    print("filtered queries:", filtered_queries)
    print("len of filtered queries:", len(filtered_queries))
    return filtered_queries if len(filtered_queries)<=3 else filtered_queries[-3:] # last elements

def getMultiQueryChain(llm):
    multi_question_template = """
    You are an AI assistant and your task is to generate three different versions of the given question to retrieve relevant documents from a vector database. By generating the multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance based similarity search.
    Provide these alternative questions separated by newlines and do not generate any extra result other than the alternate questions.
    
    Question: {question}
    """
    
    multi_question_prompt = PromptTemplate(template=multi_question_template, input_variables=["question"])
    generate_multi_queries = multi_question_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
    return generate_multi_queries

def reciprocalRankFusion(results: list[list], k=60):
    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)
    
    # No need to send scores for using docs as context
    reranked_results = [
        loads(doc)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

def getMultiQueryDocs(llm, retriever, question:str):
    print("initial question:", question)
    generate_multi_queries = getMultiQueryChain(llm=llm)
    retrieval_chain = generate_multi_queries | filterQueries | retriever.map() | reciprocalRankFusion
    return retrieval_chain.invoke(question)

def parseConversationalFusion(conversation):
    
    # print(conversation)
    user_query = conversation["data"]["question"]
    llm_answer = conversation["answer"]
    sources = set()
    
    mapper = {}
    
    for doc in conversation["data"]["context"]:
        if doc.metadata["source"] not in mapper:
            mapper[doc.metadata["source"]]=0
        mapper[doc.metadata["source"]]+=1
    
    sourceMappings = sorted(mapper.items(), key=lambda x: x[1], reverse=True)
    print(sourceMappings)
    sources = [s[0] for s in sourceMappings][:4] # top 4 sources
    print(sources)
    
    
    res = {
        "question": user_query,
        "answer": llm_answer,
        "sources": sources
    }
    return res

def getFusionLLMResponse(llm, retriever, question:str):
    validatorResponse = None
    try:
        
        validatorResponse = client.chat.completions.create(
        response_model=LLMResponse,
        messages=[
            {"role": "user", "content": f"Validate the question:  `{question}`"}
        ]
    )
    except ValidationError as ve:
        obj = LLMResponse()
        obj.question = question
        obj.isValidResponse = False
        obj.errorMessage = "Sorry, your query is not related to any of the blog contents"
        validatorResponse = obj
        print("Validation error:", str(ve))
    
    if not validatorResponse.isValidResponse:
        resp = {
            "question": question,
            "isValidResponse": validatorResponse.isValidResponse,
            "errorMessage": "Sorry, your query is not related to any of the blog contents"
        }
        return resp
    prompt_template = """
    You are a technical expert assisting with descriptive  answers based on provided tech blog content. Carefully analyze the context retrieved from the blog to answer the following question with precision. If the context includes relevant code snippets, provide them exactly as presented. When the context doesn't contain an answer, give a response based on your own knowledge. Be concise, but thorough, prioritizing accuracy from the documents. Clearly distinguish between information from the context and your own knowledge if used."

    Context: 
    {context}

    Question: 
    {question}"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    def formatDocs(docs):
        return  "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain_from_docs = (
        {
            "context": lambda x: formatDocs(x["data"]["context"]),
            "question": lambda x: x["data"]["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    rag_chain_with_source = RunnableParallel(
        {"data": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    
    res = rag_chain_with_source.invoke({
        "context": getMultiQueryDocs(llm=llm, retriever=retriever, question=question),
        "question": question
    })
    
    return parseConversationalFusion(res)
    # return res


llm = initGeminiLLM()
print("llm initiated")


vstore2 = loadVectorDB(folder_path="RAG/faissdb_1000")
print("vector store loaded")
user_question = "What is are statefulsets and volumes? Give code for same"
retriever2 = vstore2.as_retriever()
print(getFusionLLMResponse(llm=llm, retriever=retriever2, question=user_question))

# TODO: add fusion logic with gemini specific prompt first

# for vectorDB that had data other than posts
# vstore = loadVectorDB(folder_path="faissdb")
# retriever = vstore.as_retriever()

# getLLMResponse(llm=llm, retriever=retriever, question=user_question)
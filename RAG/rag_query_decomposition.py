import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import loads, dumps
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.vectorstores import FAISS    
    
load_dotenv()
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

def formatDocs(docs):
        return  "\n\n".join(doc.page_content for doc in docs)
    
def reciprocalRankFusionSources(results: list[list], k=60):
    '''
    Pass all the sources and sort in order of most occuring sources
    '''
    fused_scores = {}

    for sources in results:
        for rank, source in enumerate(sources):
            if source not in fused_scores:
                fused_scores[source] = 0
            fused_scores[source] += 1 / (rank + k)
    
    # No need to send scores for using sources as context
    reranked_results = [
        sourceUrl
        for sourceUrl, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results[:3] # return top 3 results only
    

def retrieveSubquestionsRAG(question,sub_question_generator_chain, vectorStore):
    """RAG on each sub-question"""
    
    prompt_template = """
    You are a technical expert assisting with answers based on provided tech blog content. Carefully analyze the context retrieved from the blog to answer the following question with precision. If the context includes relevant code snippets, provide them exactly as presented. When the context doesn't contain an answer, give a response based on your own knowledge. Be concise, but thorough, prioritizing accuracy from the documents. Clearly distinguish between information from the context and your own knowledge if used."

    Context: 
    {context}

    Question: 
    {question}"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    subQuestions = sub_question_generator_chain.invoke({"question":question})
    print("initial question: ", question)
    print("sub-questions: ", subQuestions)
    
    # Initialize a list to hold RAG chain results
    subqueryAnswers = []
    sources_list = []
    
    for sub_question in subQuestions:
        
        # Retrieve documents for each sub-question
        retrieved_docs = vectorStore.similarity_search(sub_question)
        print(retrieved_docs)
        sources = [d.metadata['source'] for d in retrieved_docs]
        sources_list.append(sources)
        
        # Use retrieved documents and sub-question in RAG chain
        subqueryAnswer = (prompt | llm | StrOutputParser()).invoke({"context": formatDocs(retrieved_docs), 
                                                                "question": sub_question})
        subqueryAnswers.append(subqueryAnswer)

        rerankedSourcesFiltered = reciprocalRankFusionSources(sources_list)
    # sub-ques and answer for each sub_question is returned, both as lists
    return subqueryAnswers,subQuestions, rerankedSourcesFiltered

def format_qa_pairs(questions, answers):
    """Format Q and A pairs
    to be passed as context"""
    
    print("answers lens", len(answers))
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers)):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    
    print("formatted context: ", formatted_string.strip())
    return formatted_string.strip()


def queryDecompostionRAG(question, llm, vectorStore):
    generate_multi_queries = getMultiQueryChain(llm=llm)
    sub_question_generator_chain = generate_multi_queries | filterQueries
    subqueryAnswers, subQuestions, rerankedSourcesFiltered = retrieveSubquestionsRAG(question=question, sub_question_generator_chain=sub_question_generator_chain, vectorStore=vectorStore)
    
    contextData = format_qa_pairs(questions=subQuestions, answers=subqueryAnswers)
    template = """Using the following set of Q&A pairs as context:

    {context}

    Synthesize a clear and comprehensive answer to the original question: "{question}". 
    Ensure the answer is based on the context provided, integrating relevant information from the Q&A pairs to address the main question accurately.
    """
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    queryDecompositionChain = (
    prompt
    | llm
    | StrOutputParser()
)
    llmAnswer = queryDecompositionChain.invoke({"context": contextData, "question": question})
    return {
        "question": question,
        "answer": llmAnswer,
        "sources": rerankedSourcesFiltered
    }
    

llm = initGeminiLLM()
print("llm initiated")

user_question = "What is are statefulsets and volumes? Give code for same"
vstore = loadVectorDB(folder_path="faissdb_1000")
# print("vector store loaded")

queryDecompostionRAG(question=user_question, llm=llm, vectorStore=vstore)



# getLLMResponse(llm=llm, retriever=retriever2, question=user_question)

# TODO: add fusion logic with gemini specific prompt first

# for vectorDB that had data other than posts
# vstore = loadVectorDB(folder_path="faissdb")
# retriever = vstore.as_retriever()

# getLLMResponse(llm=llm, retriever=retriever, question=user_question)
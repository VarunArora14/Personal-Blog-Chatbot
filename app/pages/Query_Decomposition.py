import streamlit as st
from langchain.vectorstores import FAISS    
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import loads, dumps
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydantic import BaseModel, AfterValidator, Field, ValidationError
from typing_extensions import Annotated
import json

st.set_page_config(
    page_title="Personal Blog Chatbot",
    page_icon="🐳",
)

client,llm,vstore = None, None, None

if "llm" not in st.session_state:
    st.error("LLM NOT IN STATE! Go to Welcome and reload!")
else:
    llm=st.session_state['llm']
    client = st.session_state['client']

if "vstore" not in st.session_state:
    st.error("VSTORE NOT IN STATE! Go to Welcome and reload!")
else:
    vstore = st.session_state["vstore"]

class LLMResponse(BaseModel):
    question: str
    isValidResponse: bool = Field(description="True if the answer is related to Kubernetes, AWS, linux, docker, networking or it's related topics, False otherwise")


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
    return reranked_results[:4] # return top 4 results only

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
        "sources": rerankedSourcesFiltered,
        "isValidResponse": True
    }
    
st.markdown("""
            
            # 🧩 Query Decomposition: Breaking Down Complex Questions 💡

            Ever asked a question that was a bit too complex? No problem! 🤔

            With **Query Decomposition**:

            - I take **multi-part or complicated queries** and break them into **smaller, manageable sub-questions** 🔨.
            - Each sub-question is answered individually, and then I piece everything back together to provide a **complete and accurate response** 🔗.

            This approach helps ensure I cover all aspects of your question—even the tough ones—by tackling it step by step! 🎯

            Try asking something complex and see how I break it down into clear, easy-to-digest answers! 🧠
            
            ---
            
            
            """)

# Initialize chat history
if "decomposition_messages" not in st.session_state:
    st.session_state.decomposition_messages = []

# Display chat messages from history on app rerun
for message in st.session_state.decomposition_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.decomposition_messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if len(st.session_state.decomposition_messages):
            response = queryDecompostionRAG(question=st.session_state.decomposition_messages[-1]["content"], llm=llm, vectorStore=vstore)
            answer = None
            if response["isValidResponse"]==False:
                answer = response["errorMessage"]
            else:
                answer = response['answer'] + "\n\nSource: " + ", ".join(x for x in response['sources'])
            st.write(answer)
    st.session_state.decomposition_messages.append({"role": "assistant", "content": answer})


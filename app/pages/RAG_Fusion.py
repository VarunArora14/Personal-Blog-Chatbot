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

st.set_page_config(
    page_title="Personal Blog Chatbot",
    page_icon="🐳",
)

llm,vstore = None, None

if "llm" not in st.session_state:
    st.write("LLM NOT IN STATE! Go to Welcome and reload!")
else:
    llm=st.session_state['llm']

if "vstore" not in st.session_state:
    st.write("VSTORE NOT IN STATE! Go to Welcome and reload!")
else:
    vstore = st.session_state["vstore"]

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
    mapper = {}
    
    for doc in conversation["data"]["context"]:
        if doc.metadata["source"] not in mapper:
            mapper[doc.metadata["source"]]=0
        mapper[doc.metadata["source"]]+=1
    
    sourceMappings = sorted(mapper.items(), key=lambda x: x[1], reverse=True)
    sources = [s[0] for s in sourceMappings][:4] # top 4 sources
    
    
    res = {
        "question": user_query,
        "answer": llm_answer,
        "sources": sources
    }
    return res

def getFusionLLMResponse(llm, retriever, question:str):
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


# st.title("🤖 Welcome to the Personal Blog Chatbot! 🌐")
### **RAG Fusion Page** 🔄


st.markdown("""
            
            # 🔄 RAG Fusion: Smart Question Expansion and Re-ranking 🧠

            With **RAG Fusion**, I go beyond a single query to make sure you get the **most accurate answer** possible! Here's how it works:

            - I take your query and **generate multiple related questions** to explore every angle of your request 💡.
            - For each question, I **retrieve relevant documents** from my knowledge sources 📄.
            - These documents are then **re-ranked** based on their relevance to the original query, ensuring the best and most accurate sources are prioritized 📊.
            - Finally, I bring everything together into a clear, well-rounded response that addresses your needs 🎯.

            By expanding the query scope and refining the document selection, RAG Fusion helps me deliver answers that are not just good, but great! 🌟
            
            ---
            
            
            """)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if len(st.session_state.messages):
            response = getFusionLLMResponse(question=st.session_state.messages[-1]["content"], llm=llm, retriever=vstore.as_retriever())
            answer = response['answer'] + "\n\nSource: " + ", ".join(x for x in response['sources'])
            st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
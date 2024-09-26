import pickle 
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS



def storeBlogDocs():
    with open('blog_urls.pkl', 'rb') as f:
        links = pickle.load(f)

    loader = AsyncHtmlLoader(links)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = loader.load_and_split(splitter)

    html2text = Html2TextTransformer()
    docsTransformed = html2text.transform_documents(docs)
    len(docsTransformed)
    
    # for i,d in enumerate(docsTransformed):
    #     if d.page_content == "\n":
    #         print(i)

    # ignore html data without any important information which have empty page content
    filteredDocs = [d for d in docsTransformed if d.page_content!='\n']
    print("len of filtered docs:", len(filteredDocs))

    with open('filtered_blog_docs.pkl', 'wb') as f:
        pickle.dump(filteredDocs, f)


# def embedDocumentsToFaiss():

def createVectorDatabase():
    # can replace with - BAAI/bge-base-en-v1.5
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    with open('filtered_blog_docs.pkl', 'rb') as f:
        filteredDocs = pickle.load(f)

    vstore = FAISS.from_documents(documents=filteredDocs, embedding=embedding_function)
    
    # save vectordb locally
    vstore.save_local(folder_path="faissdb_1000", index_name="blog")
    
    # loading method
    # vstore2 = FAISS.load_local(folder_path="faissdb", index_name="blog", embeddings=embedding_function, allow_dangerous_deserialization=True)
    # vstore2.similarity_search("What is Kubernetes?")

    

# storeBlogDocs()
# createVectorDatabase()


from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
import chainlit as cl
import os

# import os
# database_url = os.environ.get('DATABASE_URL')

# # Load environment variables from the .env file
# load_dotenv()
# api_key = os.environ.get("api_key")
# api_key = "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"
# api_key = 


# import os
# print("App running on port:", os.getenv("PORT"))
# print("Working directory:", os.getcwd())
# print("Files:", os.listdir("."))

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template =  """
    Vous êtes un assistant intéligent pour des étudiants à l'université. Répondez à la question aussi détaillée que possible en utilisant le contexte fourni. Assurez-vous de fournir tous les détails. Si la réponse n'est pas dans le contexte fourni, formulez gentiment une réponse en disant que vous ne savez pas.\n\n
	- utilisez le markdown pour la clarté de la réponse.
	- Evitez de répondre à des questions hors du sujet de l'université

	
    Contexte:\n {context}?\n
    Question: \n{question}\n

    Réponse:    
    """

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=db.as_retriever(search_kwargs={'k': 2}), return_source_documents=True, chain_type_kwargs={'prompt': prompt})
    return qa_chain

#Loading the model
def load_llm():
    llm = ChatOpenAI(model="gpt-4o",
                    #  api_key=api_key,
                     streaming=True)
    return llm

#QA Model Function
def qa_bot():
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    embeddings = OpenAIEmbeddings(
        # openai_api_key=api_key,
                                  model="text-embedding-ada-002")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

# #chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    # msg = cl.Message(content="Démarrage du Bot ...")
    # await msg.send()
    # msg.content = "Bonjour, Je sui votre assistant pour l'université d'Alger comment puis-je vous aidez ?"
    # await msg.update()
    
    cl.user_session.set("chain", chain)

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Inscription - Documents requis",
            message="Quels documents sont nécessaires pour s’inscrire ?",
             icon="/public/inscription.svg",
        ),
        cl.Starter(
            label="Logement - Options disponibles",
            message="Quelles options de logement sont disponibles ?",
             icon="/public/logement.svg",
        ),
        cl.Starter(
            label="Clubs - Adhésion",
            message="Quels clubs puis-je rejoindre ?",
             icon="/public/club.svg",
        ),
        cl.Starter(
            label="Santé - Services proposés",
            message="Quels services de santé sont proposés ?",
             icon="/public/sante.svg",
        ),
        cl.Starter(
            label="Règles d’assiduité",
            message="Quelles sont les règles d’assiduité ?",
             icon="/public/regles.svg",
        ),
        cl.Starter(
            label="Horaires des navettes",
            message="Comment accéder aux horaires des navettes ?",
             icon="/public/horaires.svg",
        ),
    ]


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["result"]

    await cl.Message(content=answer).send()


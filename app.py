import os
from dotenv import load_dotenv
from typing import TypedDict, List

from langgraph.graph import StateGraph, END

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

load_dotenv()

############################################
# LLM
############################################

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

############################################
# VECTOR DB
############################################

DB_PATH = "./vectorstore/chroma"

vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

############################################
# STATE
############################################

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    retry_count: int

############################################
# NODE 1: RETRIEVE
############################################

def retrieve(state):

    docs = retriever.invoke(state["question"])

    print("\n🔍 Retrieved:", len(docs))

    return {"documents": docs}

############################################
# NODE 2: GRADE DOCUMENTS
############################################

def grade_documents(state):

    filtered = []

    for doc in state["documents"]:

        prompt = f"""
Question:
{state['question']}

Document:
{doc.page_content}

Is this relevant? Answer yes or no.
"""

        result = llm.invoke(prompt).content.lower()

        if "yes" in result:
            filtered.append(doc)

    print("✅ Relevant Docs:", len(filtered))

    return {"documents": filtered}

############################################
# ROUTER
############################################

def route(state):

    if len(state["documents"]) == 0:
        return "rewrite"

    return "generate"

############################################
# NODE 3: REWRITE QUERY
############################################

def rewrite(state):

    prompt = f"""
Rewrite this retail inventory question
to improve retrieval quality.

Question:
{state['question']}
"""

    new_q = llm.invoke(prompt).content

    print("\n🔁 Rewritten Query:", new_q)

    return {
        "question": new_q,
        "retry_count": state["retry_count"] + 1
    }

############################################
# NODE 4: GENERATE ANSWER
############################################

def generate(state):

    context = "\n\n".join(
        [d.page_content for d in state["documents"]]
    )

    prompt = f"""
You are a Retail Inventory Fulfillment Assistant.

Use ONLY the context below.

Context:
{context}

Question:
{state['question']}

Provide:
- Inventory status
- Shipment status
- Supplier reliability
- Final fulfillment recommendation
"""

    answer = llm.invoke(prompt).content

    return {"answer": answer}

############################################
# NODE 5: SELF-RAG CHECK
############################################

def self_check(state):

    context = "\n".join(
        [d.page_content for d in state["documents"]]
    )

    prompt = f"""
Context:
{context}

Answer:
{state['answer']}

Check if answer includes:
- Inventory
- Shipments
- Supplier info

Return:
sufficient or insufficient
"""

    result = llm.invoke(prompt).content.lower()

    print("🧠 Self Check:", result)

    if "sufficient" in result:
        return "done"

    return "redo"

############################################
# LIMIT LOOP
############################################

def loop_control(state):

    if state["retry_count"] >= 2:
        return END

    return "rewrite"

############################################
# BUILD GRAPH
############################################

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_documents)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "grade")

workflow.add_conditional_edges(
    "grade",
    route,
    {
        "rewrite": "rewrite",
        "generate": "generate"
    }
)

workflow.add_edge("rewrite", "retrieve")

workflow.add_conditional_edges(
    "generate",
    self_check,
    {
        "done": END,
        "redo": "rewrite"
    }
)

app = workflow.compile()

############################################
# RUN LOOP
############################################

if __name__ == "__main__":

    print("\n🛒 Inventory Fulfillment Self-RAG Assistant\n")

    while True:

        question = input("\nAsk: ")

        result = app.invoke({
            "question": question,
            "retry_count": 0
        })

        print("\n📦 FINAL ANSWER:\n")
        print(result["answer"])

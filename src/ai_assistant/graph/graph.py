from langgraph.graph import StateGraph, END
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from src.ai_assistant.utils.prompts import load_prompt
from src.ai_assistant.graph.state import RAGState
from src.ai_assistant.rag import RAGPipeline
from src.ai_assistant.core.config import config
from src.ai_assistant.core.logger import logger


llm = ChatGoogleGenerativeAI(
    model=config.llm.model_name,
    temperature=config.llm.temperature,
    top_k=config.llm.top_k,
    top_p=config.llm.top_p,
    google_api_key=config.llm.api_key,
)

agent = create_agent(
    llm,
    middleware=[
        SummarizationMiddleware(
            llm, max_tokens_before_summary=10000, messages_to_keep=20
        ),
    ],
    system_prompt=load_prompt("system.txt"),
    checkpointer=InMemorySaver(),
)

rag_template = PromptTemplate(
    template=load_prompt("rag.txt"),
    input_variables=["context", "query"],
)

rewrite_prompt = PromptTemplate(
    template=load_prompt("rag_rewrite.txt"),
    input_variables=["query"],
)


gate_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a classifier. Decide if the user's question requires information from documents.\n"
            "If the question can be answered without documents, respond ONLY with 'NO_RAG'.\n"
            "If the question requires information from the documents, respond ONLY with 'USE_RAG'.",
        ),
        ("user", "{query}"),
    ]
)

rag_pipeline = RAGPipeline()


async def node_llm_gatekeeper(state: RAGState) -> RAGState:
    decision = await llm.ainvoke(gate_prompt.format(query=state.query))

    if decision.content.strip() == "NO_RAG":
        state.use_rag = False
        general_response = await agent.ainvoke(
            {"messages": state.query}, {"configurable": {"thread_id": "1"}}
        )
        state.answer = general_response["messages"][-1].content
    else:
        state.use_rag = True

    return state


async def node_bypass_rag(state: RAGState) -> RAGState:
    return state


async def node_optimize_query(state: RAGState) -> RAGState:
    try:
        result = await llm.ainvoke(rewrite_prompt.format(query=state.query))
        optimized = result.content.strip()

        state.query_optimized = optimized
    except Exception as e:
        state.query_optimized = state.query
        logger.error(f"Query rewrite failed: {str(e)}")

    return state

async def node_retrieve(state: RAGState) -> RAGState:
    q = state.query_optimized or state.query
    docs = await rag_pipeline.retrieve(q, 2)
    state.docs = docs
    return state


async def node_build_prompt(state: RAGState) -> RAGState:
    context = ""

    for doc in state.docs:
        context += (
            f"Document: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}\n\n"
        )

    state.prompt = rag_template.format(context=context, query=state.query)
    return state


async def node_generate(state: RAGState) -> RAGState:
    try:
        response = await agent.ainvoke(
            {"messages": state.prompt}, {"configurable": {"thread_id": "1"}}
        )
        state.answer = response["messages"][-1].content
    except Exception as e:
        state.answer = f"Error generating response: {str(e)}"
        logger.error(f"Error in node_generate: {str(e)}")

    return state


def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("gatekeeper", node_llm_gatekeeper)
    graph.add_node("optimize_query", node_optimize_query)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("build_prompt", node_build_prompt)
    graph.add_node("generate", node_generate)
    graph.add_node("bypass", node_bypass_rag)

    graph.set_entry_point("gatekeeper")

    graph.add_conditional_edges(
        "gatekeeper",
        lambda state: "rag" if state.use_rag else "no_rag",
        {
            "rag": "optimize_query",
            "no_rag": "bypass",
        },
    )

    graph.add_edge("optimize_query", "retrieve")
    graph.add_edge("retrieve", "build_prompt")
    graph.add_edge("build_prompt", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("bypass", END)

    return graph.compile()


rag_graph = build_rag_graph()

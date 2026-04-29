import boto3, os
from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, END
import time, json

# -----------------------
# Configuration (env-var injectable for App Runner)
# -----------------------
REGION = os.environ.get("REGION")
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID")
DATA_SOURCE_ID = os.environ.get("DATA_SOURCE_ID")
DATA_BUCKET = os.environ.get("DATA_BUCKET")
FEEDBACK_BUCKET = os.environ.get("FEEDBACK_BUCKET")
FLOW_ARN = os.environ.get("FLOW_ARN")

KB_NUM_RESULTS = 5
KB_OVERRIDE_SEARCH_TYPE = "SEMANTIC"
KB_CONFIDENCE_THRESHOLD = 0.7

GUARDRAIL_ID = os.environ.get("GUARDRAIL_ID")
GUARDRAIL_VERSION = os.environ.get("GUARDRAIL_VERSION", "1")

# -----------------------
# AWS Clients
# -----------------------
_session = boto3.Session(region_name=REGION)

bedrock_agent_runtime = _session.client("bedrock-agent-runtime")
comprehend = _session.client("comprehend")
a2i = _session.client("sagemaker-a2i-runtime")

llm = ChatBedrock(
    model_id="us.amazon.nova-pro-v1:0",   # us-east-1 cross-region inference profile
    region_name=REGION
)

# -----------------------
# LangGraph State
# -----------------------
class AgentState(TypedDict):
    question: str
    kb_answer: str
    sentiment: str
    final_answer: str
    confidence: float
    next_step: str
    loop_name: str

# -----------------------
# Agent 1: Supervisor / Entry Router
# -----------------------
def supervisor_router(state: AgentState):
    """Entry point for the graph – routes to KB and Sentiment agents in parallel."""
    print("Supervisor: Starting orchestration → KB & Sentiment agents.")
    # Return the initial state to be passed to the parallel nodes
    return state

# -----------------------
# Agent 2: Knowledge Base Agent
# -----------------------
def knowledge_agent(state: AgentState):
    query = state["question"]
    print(f"KB Agent: Searching knowledge base for '{query}'...")
    kb_answer = "No relevant course material found."
    confidence = 0.0
    try:
        response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": KB_NUM_RESULTS,
                    "overrideSearchType": KB_OVERRIDE_SEARCH_TYPE,
                }
            },
        )
        results = response.get("retrievalResults", [])

        if results:
            # Sort and take top result
            top = sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)[0]
            kb_answer = top.get("content", {}).get("text", "")
            confidence = float(top.get("score", 0.0))

        print(f"KB Agent: Top match score = {confidence:.2f}")

    except Exception as e:
        print(f"KB Agent Error: {e}")
        kb_answer = "Error retrieving from knowledge base."
        confidence = 0.0

    # Return only the state keys updated by this agent
    return {"kb_answer": kb_answer, "confidence": confidence}

# -----------------------
# Agent 3: Sentiment Agent
# -----------------------
def sentiment_agent(state: AgentState):
    text = state["question"]
    print("Sentiment Agent: Analyzing sentiment...")
    sentiment = "NEUTRAL"
    try:
        response = comprehend.detect_sentiment(Text=text, LanguageCode="en")
        sentiment = response.get("Sentiment", "NEUTRAL")
        print(f"Sentiment Agent: Detected '{sentiment}'")
    except Exception as e:
        print("Sentiment Agent Error:", e)
        sentiment = "NEUTRAL"

    # Return only the state keys updated by this agent
    return {"sentiment": sentiment}

# -----------------------
# Agent 4: Join Node
# -----------------------
def join_results(state: AgentState):
    """Synchronizes after KB and Sentiment complete."""
    print("Join Node: Both KB and Sentiment agents finished.")
    # LangGraph automatically merges the state updates from the parallel branches
    # based on the keys returned by each agent.
    return state

# -----------------------
# Agent 5: LLM Final Answer Generator  (with guardrails)
# -----------------------
def generate_final_answer(state: AgentState):
    print("Final Generator: Synthesizing final response with LLM...")

    # Confidence floor: don't let the LLM improvise when KB has nothing useful
    if state["kb_answer"] in ("No relevant course material found.", "Error retrieving from knowledge base."):
        print("Final Generator: KB answer is empty — skipping LLM to avoid hallucination, escalating.")
        return {"final_answer": (
            "I don't have enough information in the course materials to answer this accurately. "
            "I'm escalating your question to the course TAs for a reliable response."
        )}

    # Hardened prompt: constrain the LLM to the KB source only
    prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert academic assistant for the 'Machine Learning Introduction' course at Ben-Gurion University. "
     "Your goal is to provide deep, professional, and accurate answers based on the provided Course Material Excerpt. "
     
     "### INSTRUCTION TUNING RULES: "
     "1. **Semantic Flexibility**: Terminology may vary. If a student asks about a 'technique' or 'method', relate it to the 'algorithms' in the text (e.g., 'Memorize algorithm'). "
     "2. **Logical Inference**: If the excerpt mentions a concept briefly (e.g., 'Underfitting is related to approximation error'), use your expertise to explain that connection clearly based on the context. "
     "3. **Mathematical Precision**: Use LaTeX for all formulas and notations. For example, use $err(h, D)$ for error or $m$ for sample size, as seen in the materials[cite: 85, 203]. "
     "4. **Depth over Safety**: Do not be overly restrictive. If the excerpt contains the core answer, expand on it professionally. Only escalate to TAs if the excerpt is truly irrelevant to the topic. "
     "5. **Response Format**: Structure your answer with clear headings and bold terms. "
     "6. **Strict Refusals**: If the Student Question is completely unrelated to the Course Material Excerpt (e.g., asking for a recipe), you must output EXACTLY: 'I couldn't find a reliable answer in the provided course materials.' Do not add any formatting, markdown, or explanations."),
    
    ("human",
     "Student Question: {question}\n"
     "Detected Sentiment: {sentiment}\n"
     "Course Material Excerpt: {kb_answer}\n\n"
     "Using the excerpt provided, write a comprehensive academic response. ")
])
    messages = prompt.format_messages(
        question=state["question"],
        sentiment=state["sentiment"],
        kb_answer=state["kb_answer"]
    )

    final_answer = "[Error generating response]"
    try:
        # Attach guardrail — contextual grounding checks response is grounded in the
        # kb_answer, which is embedded in the conversation messages above
        llm_with_guardrail = llm.bind(
            guardrailConfig={
                "guardrailIdentifier": GUARDRAIL_ID,
                "guardrailVersion": GUARDRAIL_VERSION,
                "trace": "enabled",
            }
        )
        response = llm_with_guardrail.invoke(messages)
        final_answer = response.content

        # Detect if the guardrail blocked the output
        if not final_answer or final_answer.strip() == "":
            print("Final Generator: Guardrail blocked the response.")
            final_answer = (
                "I wasn't able to generate a reliable answer based on the provided course materials. "
                "Please ask the teaching staff or instructor for assistance."
            )
        else:
            print(f"Final Generator: Response generated ({len(final_answer)} chars).")

    except Exception as e:
        print(f"LLM Error: {e}")

    return {"final_answer": final_answer}


# -----------------------
# Agent 6: Human Escalation (A2I) + Feedback Loop
# -----------------------
def human_agent(state: AgentState):
    print("Human Agent: Escalating to A2I...")
    loop_name = f"loop-{int(time.time())}"
    final_answer = (
        "Your question has been escalated to the course TAs. "
        "Once they provide an answer, it will be automatically added to the course knowledge base for future reference."
    )
    try:
        a2i.start_human_loop(
            HumanLoopName=loop_name,
            FlowDefinitionArn=FLOW_ARN,
            HumanLoopInput={"InputContent": json.dumps({
                "question": state["question"],
                "material_suggestion": state["kb_answer"]
            })},
        )
        print(f"  Escalation started → loop: {loop_name}")
        print(f"  Answer will be processed automatically via EventBridge + Lambda.")

    except Exception as e:
        print(f"Human Agent Error: {e}")
        loop_name = ""

    return {"final_answer": final_answer, "loop_name": loop_name}


# -----------------------
# Router: Confidence-based Branching
# -----------------------
def confidence_router(state: AgentState):
    score = state.get("confidence", 0.0)
    print(f"Router: Checking confidence = {score:.2f}")
    if score >= KB_CONFIDENCE_THRESHOLD:
        print("Router: High confidence → route to LLM response")
        return "generator"
    else:
        print("Router: Low confidence → escalate to human review")
        return "human"

# -----------------------
# Build LangGraph
# -----------------------
graph = StateGraph(AgentState)

graph.add_node("supervisor", supervisor_router)
graph.add_node("kb", knowledge_agent)
graph.add_node("sentiment", sentiment_agent)
graph.add_node("join_results", join_results)
graph.add_node("generator", generate_final_answer)
graph.add_node("human", human_agent)

graph.set_entry_point("supervisor")

# Parallel branches
graph.add_edge("supervisor", "kb")
graph.add_edge("supervisor", "sentiment")

# Synchronize
graph.add_edge("kb", "join_results")
graph.add_edge("sentiment", "join_results")

# Conditional route
graph.add_conditional_edges("join_results", confidence_router, {
    "generator": "generator",
    "human": "human"
})

graph.add_edge("generator", END)
graph.add_edge("human", END)

app = graph.compile()
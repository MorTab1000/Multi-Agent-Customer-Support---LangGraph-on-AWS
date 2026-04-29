import os
import boto3
from fastapi import FastAPI
from pydantic import BaseModel
from app.main import (
    app as graph_app,
    REGION,
    KB_CONFIDENCE_THRESHOLD,
)

api = FastAPI(title="Academic Assistant API")

_session = boto3.Session(
    profile_name=os.environ.get("AWS_PROFILE") or None,
    region_name=REGION,
)
_bedrock = _session.client("bedrock-runtime")

OFF_TOPIC_MSG = (
    "I can only answer questions related to the 'Machine Learning Introduction' course lectures. "
    "Please ask a relevant question."
)

_DOMAIN_SYSTEM = (
    "You are a domain classifier for a course called 'Machine Learning Introduction'. "
    "Decide if the user question is related to machine learning concepts. "
    "CRITICAL INSTRUCTION: Be semantically flexible. If a user asks about 'learning techniques', "
    "'memorization', 'methods', or 'algorithms', you MUST treat it as a machine learning concept "
    "(e.g., algorithmic memorization vs. generalization). "
    "Topics include supervised/unsupervised learning, overfitting, underfitting, evaluation metrics, and algorithms. "
    "If the user asks about completely unrelated domains (e.g., finance, politics, sports, general chit-chat), answer NO. "
    "Reply with exactly one word: YES if related, NO if not."
)


def _is_in_domain(question: str) -> bool:
    """Returns True if the question is about the Machine Learning Introduction course."""
    try:
        response = _bedrock.converse(
            modelId="us.amazon.nova-pro-v1:0",
            system=[{"text": _DOMAIN_SYSTEM}],
            messages=[{"role": "user", "content": [{"text": question}]}],
            inferenceConfig={"maxTokens": 5, "temperature": 0},
        )
        verdict = response["output"]["message"]["content"][0]["text"].strip().upper()
        print(f"Domain check: '{question[:60]}' → {verdict}", flush=True)
        return verdict.startswith("YES")
    except Exception as e:
        print(f"Domain check error: {e}", flush=True)
        return True  # fail open


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    confidence: float
    escalated: bool


@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not _is_in_domain(req.question):
        return AskResponse(answer=OFF_TOPIC_MSG, confidence=0.0, escalated=False)

    result = graph_app.invoke({"question": req.question})
    escalated = result.get("confidence", 1.0) < KB_CONFIDENCE_THRESHOLD
    return AskResponse(
        answer=result.get("final_answer", ""),
        confidence=result.get("confidence", 0.0),
        escalated=escalated,
    )
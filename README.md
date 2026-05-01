# Multi-Agent Academic Assistant - AWS Cloud Integration (LangGraph)

🟢 Interactive Architecture Walkthrough: [Live Demo](https://mortab1000.github.io/multi-agent-support-langgraph/) - Click to see the step-by-step data flow.

A production-ready multi-agent pipeline that answers Machine Learning Introduction course questions using LangGraph, Amazon Bedrock, Amazon Comprehend, and SageMaker A2I.

Built as a hands-on workshop lab demonstrating real-world multi-agent architecture on AWS.

![Architecture](docs/architecture.png)

## Architecture & Integration Flow

```text
POST /ask → Domain Classifier (api.py) → LangGraph Graph
                                              ├── KB Agent (Bedrock Knowledge Base)
                                              ├── Sentiment Agent (Amazon Comprehend)
                                              └── Join → Confidence Router
                                                              ├── ≥ 0.75 → LLM Generator (Nova Pro + Guardrail v4)
                                                              └── < 0.75  → Human Escalation (SageMaker A2I)
```

### Key Cloud & Infrastructure Integrations:

- **Serverless Deployment:** The LangGraph API is containerized using Docker and deployed securely on AWS App Runner.
- **Human-in-the-Loop (HITL):** Integrated SageMaker A2I to catch low-confidence LLM outputs and route them to human reviewers.
- **Automated Feedback Loop:** Configured an event-driven architecture using EventBridge and Lambda (`a2i_completion_handler.py`). When a human answers an escalated ticket, the system automatically writes it back to S3 and triggers a Bedrock Knowledge Base sync to improve future answers.
- **LLM is skipped entirely** when the KB has no relevant answer (prevents hallucination)
- **Infrastructure as Code (IaC)**: Utilized CloudFormation templates for provisioning S3 buckets, IAM roles, ECR, and Lambda.

## Project Structure

<details>
<summary><b>📂 Click to view the full directory tree</b></summary>

```text
├── app/
│   ├── api.py                    # FastAPI wrapper + domain classifier
│   └── main.py                   # LangGraph graph architecture
├── data/
│   └── pdfs/                     # Lecture note PDFs for Bedrock Knowledge Base
├── infra/
│   ├── cloudformation.yaml       # IaC for S3, IAM, ECR, EventBridge, Lambda
│   └── a2i_worker_template.xml   # SageMaker A2I reviewer UI template
├── lambda/
│   └── a2i_completion_handler.py # Feedback loop logic: A2I → S3 → KB sync
├── scripts/
│   ├── deploy.sh                 # Fully automated deployment script
│   └── destroy.sh                # Full cloud teardown script
├── ui/
│   ├── streamlit_app.py          # Streamlit chat frontend for the /ask API
│   └── requirements.txt          # UI-specific Python dependencies
├── Dockerfile                    # Container configuration (python:3.12-slim)
├── mcp_server.py                 # MCP Server: Bridge for Claude Desktop/Cursor tools  
└── requirements.txt
```
</details>

## Quick Start

### Prerequisites

- AWS account with permissions for Bedrock, S3, IAM, ECR, App Runner, Comprehend, SageMaker
- AWS CLI configured (`aws configure` or named profile)
- Docker (for App Runner deployment)
- Python 3.12+

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Deploy

The deployment process is automated via bash scripts that handle CloudFormation execution, Docker image building, ECR pushing, and App Runner deployment.

```bash
# Full automated deployment (CloudFormation → KB → Guardrail → Docker → ECR → App Runner)
bash scripts/deploy.sh
```

The script will print the live App Runner URL when complete.

### 3. Test the API

```bash
curl -s -X POST <YOUR_APP_RUNNER_URL>/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is overfitting in machine learning?"}' | jq
```

### 4. Tear down infrastructure

```bash
bash scripts/destroy.sh
```

## AWS Resources Used


| Service                        | Purpose                                                   |
| ------------------------------ | --------------------------------------------------------- |
| Amazon Bedrock (Nova Pro)      | LLM response generation                                   |
| Amazon Bedrock Knowledge Bases | Course material retrieval (Titan Embeddings + S3 Vectors) |
| Amazon Bedrock Guardrails      | Content filtering + contextual grounding                  |
| Amazon Comprehend              | Sentiment analysis                                        |
| SageMaker A2I                  | Human escalation review                                   |
| AWS App Runner                 | Serverless container hosting                              |
| Amazon ECR                     | Docker image registry                                     |
| AWS Lambda + EventBridge       | A2I feedback loop automation                              |
| Amazon S3                      | Course material data + feedback storage                   |


## MCP Integration

The project now supports Model Context Protocol (MCP), allowing AI agents (like Claude or Cursor) to directly query the AWS Knowledge Base as a tool.

## Streamlit Chat UI

You can run a modern chat frontend locally that calls the deployed `/ask` endpoint.

Live Streamlit deployment:
[see demo app](https://multi-agent-support-langgraph-ux366hqvhua2vwrcfqmwh2.streamlit.app/)

### Configure API URL

To connect the UI to your backend API, you need to set the `SUPPORT_API_URL` environment variable.

**For Local Development:**
Create a `.env` file from `.env.example` and set your deployed App Runner URL:

```bash
SUPPORT_API_URL=https://46r8ga4hcc.us-east-1.awsapprunner.com/ask
```

**For Cloud Deployment:**
If deploying to Streamlit Community Cloud (like the live demo), set this URL securely via Streamlit Secrets in the app's advanced settings.

### Run

```bash
streamlit run ui/streamlit_app.py
```

Features included:

- Chat-style interface with message history via `st.session_state`
- Custom assistant icons: Robot (`🤖`) for AI and Teacher (`🧑‍🏫`) for escalated responses
- Escalation indicator box when API returns `escalated: true`
- Confidence score caption for each assistant response
- Robust network/HTTP/JSON error handling
- LaTeX formula rendering in responses through Streamlit Markdown (`$...$`, `$$...$$`)

### Project Evolution & My Contributions

> This project originated from a foundational LangGraph & AWS workshop. However, it has been significantly expanded, refactored, and transformed into a full-stack, production-ready application. 
>
> **Key enhancements and custom implementations include:**
>
> - **Domain Shift & Real-World Use Case:** Adapted the base logic to function as an Academic Assistant, processing real-world university lecture slides and materials.
> - **Full-Stack UI:** Developed and integrated a persistent, user-friendly Streamlit frontend.
> - **MCP Server Integration:** Engineered a Model Context Protocol (MCP) server, allowing external AI agents (like Claude Desktop or Cursor) to query the AWS Knowledge Base directly.
> - **Cloud Infrastructure & CI/CD:** Architected the complete serverless deployment (App Runner, EventBridge, Lambda, SageMaker A2I) and wrote the automated bash deployment/teardown scripts.
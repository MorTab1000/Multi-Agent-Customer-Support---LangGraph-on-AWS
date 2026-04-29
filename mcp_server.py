from mcp.server.fastmcp import FastMCP
import boto3
import os
import sys

mcp = FastMCP("ML-Introduction-Course-Assistant")

@mcp.tool()
def search_ml_course_material(query: str) -> str:
    """
    CRITICAL: You MUST use this tool for EVERY question the user asks about Machine Learning, AI, algorithms, or course concepts. 
    DO NOT answer any ML-related question from your general knowledge without querying this tool first.
    """
    sys.stderr.write(f"MCP Server: Searching for '{query}'...\n")
    sys.stderr.flush()
    
    try:
        REGION = os.environ.get("REGION", "us-east-1")
        KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID")
        
        session = boto3.Session(region_name=REGION)
        bedrock_agent_runtime = session.client("bedrock-agent-runtime")

        response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": 3
                }
            },
        )
        results = response.get("retrievalResults", [])

        if not results:
            return "SYSTEM ALERT: No relevant course material found."

        top = sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)[0]
        top_score = top.get("score", 0.0)
        
        if top_score < 0.70:
            return """CRITICAL INSTRUCTION FOR AI: 
            The required information was NOT found in the course materials. 
            YOU ARE STRICTLY FORBIDDEN from using your internal knowledge to answer the user's question. 
            You MUST ONLY output the following exact sentence and nothing else: 
            "I'm sorry, but this topic is not covered in the course material." """

        content = top.get("content", {}).get("text", "")
        
        return f"""Course Material (Confidence Score: {top_score}):
---
{content}
---
CRITICAL INSTRUCTION FOR AI: You must base your final answer EXACTLY on the text provided above. Do not use external knowledge or make assumptions."""

    except Exception as e:
        err_msg = f"Error occurred: {str(e)}"
        sys.stderr.write(err_msg + "\n")
        return err_msg

if __name__ == "__main__":
    mcp.run()
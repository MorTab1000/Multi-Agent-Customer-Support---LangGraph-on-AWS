from mcp.server.fastmcp import FastMCP
import boto3
import os

mcp = FastMCP("ML-Introduction-Course-Assistant")

REGION = os.environ.get("REGION", "us-east-1")
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID")

session = boto3.Session(region_name=REGION)
bedrock_agent_runtime = session.client("bedrock-agent-runtime")

@mcp.tool()
def search_ml_course_material(query: str) -> str:
    """
    Search for relevant information in the Machine Learning Introduction course knowledge base
    and return the most relevant answer.
    @query: The student's question to search for in the course knowledge base.
    @return: The most relevant answer from the course knowledge base, or an error message.
    Important: Use only information from the knowledge base. Do not use external assumptions.
    If the answer is not found, return "No relevant course material found."
    """
    print(f"MCP Server: Searching for '{query}'...")
    
    try:
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
            return "No relevant course material found."

        top = sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)[0]
        content = top.get("content", {}).get("text", "")
        
        return content

    except Exception as e:
        return f"Error occurred while accessing the course knowledge base: {str(e)}"

if __name__ == "__main__":
    mcp.run()
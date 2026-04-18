from mcp.server.fastmcp import FastMCP
import boto3
import os

mcp = FastMCP("AWS-FAQ-Service")

REGION = os.environ.get("REGION", "us-east-1")
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID")

session = boto3.Session(region_name=REGION)
bedrock_agent_runtime = session.client("bedrock-agent-runtime")

@mcp.tool()
def search_leumi_trade_faq(query: str) -> str:
    """
    search for relevant information in the Leumi Trade FAQ knowledge base and return the most relevant answer.
    @query: The user's question or query to search for in the FAQ knowledge base.
    @return: The most relevant answer from the FAQ knowledge base, or an error message if
    Important: Use only the information from the knowledge base to answer the question. Do not use any external information or assumptions. If the answer is not found in the knowledge base, return "Results not found in the knowledge base."
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
            return "The information you are looking for was not found in the Leumi Trade FAQ."

        top = sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)[0]
        content = top.get("content", {}).get("text", "")
        
        return content

    except Exception as e:
        return f"Error occurred while accessing the knowledge base: {str(e)}"

if __name__ == "__main__":
    mcp.run()
import json
import os
import boto3

s3 = boto3.client("s3")
bedrock_agent = boto3.client("bedrock-agent")

DATA_BUCKET = os.environ["DATA_BUCKET"]
KNOWLEDGE_BASE_ID = os.environ["KNOWLEDGE_BASE_ID"]
DATA_SOURCE_ID = os.environ["DATA_SOURCE_ID"]


def _parse_s3_uri(uri: str):
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri!r}")
    parts = uri.replace("s3://", "", 1).split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"S3 URI must include bucket and key: {uri!r}")
    return parts[0], parts[1]


def handler(event, context):
    try:
        detail = event["detail"]
        loop_name = detail["humanLoopName"]
        output_s3_uri = detail["humanLoopOutput"]["outputS3Uri"]
    except (TypeError, KeyError) as e:
        print(f"Invalid event payload: {e}; event={event!r}")
        return {"statusCode": 400, "error": "Invalid A2I completion event payload"}

    print(f"Processing completed loop: {loop_name}")
    print(f"Output URI: {output_s3_uri}")

    # Read A2I output from S3
    try:
        bucket, key = _parse_s3_uri(output_s3_uri)
        resp = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(resp["Body"].read())

        question = data.get("inputContent", {}).get("question")
        human_answers = data.get("humanAnswers") or []
        answer = (
            human_answers[0]
            .get("answerContent", {})
            .get("human_response")
            if human_answers
            else None
        )
        if not question or not answer:
            raise ValueError("Missing question or human_response in A2I output payload")
    except Exception as e:
        print(f"Failed processing A2I output: {e}")
        return {"statusCode": 400, "loopName": loop_name, "error": "Invalid A2I output payload"}

    # Write TA note to data bucket
    note_key = f"ta_note_{loop_name}.txt"
    content = f"Q: {question}\nA: {answer}"
    s3.put_object(Bucket=DATA_BUCKET, Key=note_key, Body=content.encode())
    print(f"TA note written to s3://{DATA_BUCKET}/{note_key}")

    # Start KB ingestion
    resp = bedrock_agent.start_ingestion_job(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        dataSourceId=DATA_SOURCE_ID,
    )
    job_id = resp["ingestionJob"]["ingestionJobId"]
    print(f"KB ingestion started: {job_id}")

    return {"statusCode": 200, "loopName": loop_name, "ingestionJobId": job_id}

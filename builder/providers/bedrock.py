# builder/providers/bedrock.py
import json
import time
import boto3
from typing import Dict, Any, List

bedrock = boto3.client("bedrock-agent")  # 'bedrock-agent' for agents APIs


def create_or_update_agent(spec: Dict[str, Any]) -> str:
    """
    spec keys: name, instruction, model_id, role_arn, guardrail_arn?, kb_ids?, action_groups?
    returns agent_id
    """
    # Try find existing
    agents = boto3.client("bedrock-agent").list_agents()["agents"]
    agent = next((a for a in agents if a["agentName"] == spec["name"]), None)

    if not agent:
        resp = bedrock.create_agent(
            agentName=spec["name"],
            instruction=spec["instruction"],
            foundationModel=spec["model_id"],
            agentResourceRoleArn=spec["role_arn"],
            guardrailConfiguration={"guardrailIdentifier": spec["guardrail_arn"]} if spec.get(
                "guardrail_arn") else None,
            autoPrepare=True  # let Bedrock keep DRAFT in sync while iterating
        )
        agent_id = resp["agent"]["agentId"]
    else:
        agent_id = agent["agentId"]
        bedrock.update_agent(
            agentId=agent_id,
            instruction=spec["instruction"],
            foundationModel=spec["model_id"],
            agentResourceRoleArn=spec["role_arn"],
        )

    # Knowledge bases
    if spec.get("kb_ids"):
        for kb_id in spec["kb_ids"]:
            bedrock.associate_agent_knowledge_base(
                agentId=agent_id,
                knowledgeBaseId=kb_id
            )

    # Action groups (Lambda or OpenAPI schema)
    if spec.get("action_groups"):
        for ag in spec["action_groups"]:
            bedrock.create_agent_action_group(
                agentId=agent_id,
                agentVersion="DRAFT",
                actionGroupName=ag["name"],
                description=ag.get("description", ""),
                actionGroupExecutor={"lambda": {
                    "lambdaArn": ag["lambda_arn"]}} if "lambda_arn" in ag else {},
                apiSchema={"s3": {
                    "s3BucketName": ag["s3_bucket"], "s3ObjectKey": ag["s3_key"]}} if "s3_bucket" in ag else {},
                parentActionGroupSignature="AMAZON.UserInput" if ag.get(
                    "user_input", False) else None,
            )

    # Prepare draft
    bedrock.prepare_agent(agentId=agent_id)
    return agent_id

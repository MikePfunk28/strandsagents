Short answer: we can make your Strands “agent-that-builds-agents” fully ownable, source-verified, and production-deployable to AWS (Bedrock) or local (Ollama) by adding (1) a meta-decorator @agent that compiles a spec → files, (2) an interactive builder CLI that thinks → verifies (MCP) → acts → logs (OTel) for each step, and (3) deploy targets: CloudFormation (Bedrock agents) and EC2/Ollama (for local models), plus a small green/black frontend.

Below is a tight, working blueprint with code you can drop into a new folder (keeps your strands-meta intact). I cite AWS/MCP/Ollama/OTel docs for every moving part.

Viability (✅)

Bedrock agents are first-class IaC via AWS::Bedrock::Agent CloudFormation, including ActionGroups (Lambda/OpenAPI), KnowledgeBases, Guardrails, etc. So we can generate a single CFN and deploy the agent reliably.


Programmatically, we can also call CreateAgent / CreateAgentActionGroup / AssociateAgentKnowledgeBase / PrepareAgent from boto3.



Ollama exposes a clean REST API for generate/chat, so our generated Python agent can talk to http://localhost:11434.



MCP is the right way to “look things up to verify” before each action. We can ship MCP clients/servers and wire them to our builder.



OpenTelemetry (OTel) can instrument our builder + generated agents for traces/logs.


Conclusion: 95%+ confidence to implement end-to-end.
```bash
strands-agent-factory/
  builder/
    __init__.py
    decorators.py         # @tool and new @agent meta-decorator
    builder_cli.py        # interactive “think → verify (MCP) → act → log”
    mcp_verify.py         # MCP calls to fetch/confirm docs before each step
    otel_setup.py         # OTel tracing+logging
    providers/
      bedrock.py          # boto3 create/update/prepare logic
      ollama.py           # http client to localhost:11434
    templates/
      bedrock_agent.cfn.yaml   # CloudFormation for Bedrock Agent + IAM
      ollama_stack.cfn.yaml    # Optional: EC2+SSM+SG to run Ollama remotely
  generated/              # per-agent outputs here (new folder, not strands-meta)
    <agent_slug>/
      deploy/             # CFN + params
      agent/
        main.py           # runnable Python agent (Ollama or Bedrock)
        tools/            # custom tool stubs
      logs/
  frontend/
    app/ (React/Tailwind; green/black theme)
    ...
  README.md
```

Frontend (green/black)

Use a tiny React/Tailwind app (think Kiro UX but green). It just collects the wizard answers and calls the CLI via a local HTTP bridge or Node child-process. (Omitted code for brevity; happy to drop a production-ready page if you want it now.)

Security & AWS specifics to get right

Least-privilege IAM for the Agent role: allow only specific Lambda ARNs; restrict logs. (Policy stub above—tighten for your account.) AgentResourceRoleArn is mandatory.


Lambda resource-based policy must allow bedrock.amazonaws.com to invoke the function(s).


Guardrails: default to enabled when user opts in; manageable via API or CFN (Guardrail/GuardrailVersion).


Knowledge base: associate only when the index is Enabled; the API/CLI & docs are explicit on this.


Prepare/AutoPrepare: either call PrepareAgent after changes or set AutoPrepare: true in CFN (we did) to keep DRAFT updated.


Why this “owns the flow” (and is better than your current script)

Separation of concerns: the meta-decorator (@agent) captures intent; the builder compiles that intent into code + IaC.

Deterministic pipeline: each step plans → verifies (MCP, official docs) → acts; all steps are OTel-logged with rationale summaries so you can audit “is it thinking correctly?”

AWS-native deploy: single CloudFormation for agents (ActionGroups/KB/Guardrails) gives you repeatable infra, not ad-hoc clicks.


Provider-toggle: same UX yields Bedrock or Ollama agents (your default can be “Bedrock + Anthropic Claude” via the FoundationModel property; use the model ID your account/region supports).


Future-proof: if you want AgentCore later (gateway/memory/obs), add it as an alternate deploy target; AWS has docs & samples.


Next steps (fast)

Drop these files into strands-agent-factory/ and run:

python -m builder.builder_cli --interactive build


Choose Bedrock → review generated/<agent>/deploy/bedrock_agent.cfn.yaml; deploy it via aws cloudformation deploy with your params (AgentName, Instruction ≥40 chars, FoundationModel, Lambda ARN, etc.). CFN resource type reference here.


Choose Ollama → run ollama serve and your generated/<agent>/agent/main.py. (API refs above.)


Wire your existing MCP servers into mcp_verify.py to fetch & summarize docs before each step (e.g., Bedrock API pages, Lambda policy page, model list).

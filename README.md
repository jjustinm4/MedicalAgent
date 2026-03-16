# MedicalAgent: Local Agentic Medical Image Interpreter

This project is a **fully local learning platform** for agentic programming with:

- local LLM reasoning using **Gemma via Ollama**
- **LangGraph** multi-agent orchestration
- **TensorFlow** pretrained CNN tool
- **BLIP (Hugging Face)** vision-language tool
- **DuckDuckGo** web search tool
- **Streamlit** UI

It is designed for experimenting with **tool calling, shared memory, reflection loops, and dynamic routing**.

## Important Note

This is an educational engineering project and **not a medical diagnostic system**.

## Architecture

```text
Streamlit UI
	|
	v
LangGraph Orchestration
	|
	+--> Gemma (Ollama HTTP: localhost:11434)
	+--> CNN Tool (TensorFlow, in-process)
	+--> VLM Tool (BLIP, in-process)
	+--> Research Tool (DuckDuckGo)
```

### LangGraph Nodes

- `PlannerAgent`
- `ImageDecisionAgent`
- `CNNToolNode`
- `VLMToolNode`
- `ResearchAgent`
- `CriticAgent`
- `FinalResponseAgent`

The graph includes a reflection loop from `CriticAgent` back to `ImageDecisionAgent` for retries.

## Project Structure

```text
MedicalAgent/
  app.py
  requirements.txt
  .env.example
  src/medical_agent/
	config.py
	llm.py
	state.py
	agents/nodes.py
	graph/workflow.py
	tools/cnn_tool.py
	tools/vlm_tool.py
	tools/search_tool.py
```

## Prerequisites

1. Python 3.10+
2. Ollama running locally
3. Gemma model pulled in Ollama

Example Ollama commands:

```bash
ollama pull gemma:2b
ollama run gemma:2b
```

## Setup

From the `MedicalAgent` folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional environment variables:

```bash
set OLLAMA_BASE_URL=http://localhost:11434
set OLLAMA_MODEL=gemma:2b
set CRITIC_CONFIDENCE_THRESHOLD=0.65
set MAX_RETRY_LOOPS=2
set BLIP_CAPTION_MODEL=Salesforce/blip-image-captioning-base
set BLIP_VQA_MODEL=Salesforce/blip-vqa-base
```

## Run

```bash
streamlit run app.py
```

Then open the local Streamlit URL in your browser.

## First-Run Behavior

- TensorFlow and BLIP pretrained weights are downloaded automatically on first use.
- Subsequent runs use local cache and are faster.

## How It Demonstrates Agentic Programming

- **PlannerAgent** interprets user goal and sets strategy.
- **ImageDecisionAgent** dynamically selects the next tool.
- **Tool nodes** run CNN / VLM / Research as callable capabilities.
- **CriticAgent** reflects on confidence and can trigger retries.
- **FinalResponseAgent** synthesizes a single user-facing explanation.

All agents read and write a shared memory state in LangGraph.

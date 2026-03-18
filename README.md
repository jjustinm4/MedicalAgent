# MedicalAgent: Local Agentic Medical Image Interpreter

This project is a learning platform for agentic programming with:

- LLM reasoning using **Google Gemini** with optional **Ollama fallback**
- **LangGraph** multi-agent orchestration
- **Chest X-ray classifier (Hugging Face/torch)** with generic fallback
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
	+--> Gemini API (primary)
	+--> Ollama Gemma (fallback, optional)
	+--> CNN Tool (medical chest X-ray classifier, in-process)
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
2. A valid Google Gemini API key
3. (Optional) Ollama running locally with Gemma for fallback

## Setup

From the `MedicalAgent` folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional environment variables:

```bash
set GEMINI_API_KEY=your_api_key_here
set GEMINI_MODEL=gemini-2.0-flash
set GEMINI_BASE_URL=https://generativelanguage.googleapis.com
set OLLAMA_FALLBACK_ENABLED=true
set OLLAMA_BASE_URL=http://localhost:11434
set OLLAMA_MODEL=gemma3:4b
set CRITIC_CONFIDENCE_THRESHOLD=0.65
set MAX_RETRY_LOOPS=2
set MEDICAL_AGENT_LOG_LEVEL=INFO
set CHEST_XRAY_MODEL=dima806/chest_xray_pneumonia_detection
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

# MedicalAgent: Local Agentic Medical Image Interpreter

This project is a learning platform for agentic programming with:

- LLM reasoning using **Google Gemini** with optional **Ollama fallback**
- **LangGraph** multi-agent orchestration
- **Chest X-ray classifier (Hugging Face/torch)** with generic fallback
- **BLIP (Hugging Face)** vision-language tool
- **DuckDuckGo** web search tool
- **Streamlit** UI

1. `cd MedicalAgent`
2. Create and activate a Python environment
3. `pip install -r requirements.txt`
4. Set `GEMINI_API_KEY` in `.env` or your shell (optional: enable Ollama fallback with `OLLAMA_*` vars)
5. `streamlit run app.py`

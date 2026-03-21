import sys
sys.path.insert(0, "e:\\study\\MedicalAgent\\MedicalAgent\\src")
from medical_agent.config import load_settings
from medical_agent.graph.workflow import run_workflow
import json
import logging

logging.basicConfig(level=logging.INFO, force=True)

config = load_settings()

print("=" * 80)
print("WORKFLOW TEST: 'give detailed description' query")
print("=" * 80)
print(f"Config: Ollama fallback={config.ollama_fallback_enabled}, Gemini model={config.gemini_model}")
print()

# Build workflow with resilient LLM client
print("=" * 80)
print("EXECUTING WORKFLOW...")
print("=" * 80 + "\n")

try:
    result = run_workflow(
        image_path=r"e:\study\MedicalAgent\MedicalAgent\sample_images\IM-0019-0001.jpeg",
        user_query="give detailed description"
    )
    print("\n" + "=" * 80)
    print("FINAL RESULT:")
    print("=" * 80)
    print(json.dumps(result, indent=2, default=str))
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[OK] Workflow completed successfully")

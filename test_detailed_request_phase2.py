import sys
sys.path.insert(0, "e:\\study\\MedicalAgent\\MedicalAgent\\src")
from medical_agent.config import load_settings
from medical_agent.graph.workflow import run_workflow
import json
import logging

logging.basicConfig(level=logging.INFO, force=True)

config = load_settings()

print("=" * 80)
print("WORKFLOW TEST PHASE 2: Clarification answers + Research routing")
print("=" * 80)
print(f"Config: Ollama fallback={config.ollama_fallback_enabled}, Gemini model={config.gemini_model}")
print()

# First pass: "give detailed description" → triggers clarification
print("=" * 80)
print("PHASE 2A: Initial query triggers clarification agent")
print("=" * 80 + "\n")

try:
    result1 = run_workflow(
        image_path=r"e:\study\MedicalAgent\MedicalAgent\sample_images\IM-0019-0001.jpeg",
        user_query="give detailed description"
    )
    
    print("\nPhase 2A Result:")
    print(f"  - Next node: {result1.get('next_node')}")
    print(f"  - Needs clarification: {result1.get('needs_clarification')}")
    print(f"  - Clarification questions:")
    for q in result1.get('clarification_questions', []):
        print(f"    • {q}")
        
except Exception as e:
    print(f"ERROR in Phase 2A: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Second pass: Submit clarification answers
print("\n" + "=" * 80)
print("PHASE 2B: Re-run with clarification answers (triggering research)")
print("=" * 80 + "\n")

# Simulate clarification answers
clarified_query = """
give detailed description

[CLARIFICATION ANSWERS]
Q1: Yes, I'm asking about the entire image
Q2: I want to know if there are abnormalities detected
Q3: This is a chest X-ray image
"""

try:
    result2 = run_workflow(
        image_path=r"e:\study\MedicalAgent\MedicalAgent\sample_images\IM-0019-0001.jpeg",
        user_query=clarified_query
    )
    
    print("\nPhase 2B Result (with clarification):")
    print(f"  - Analysis type: {result2.get('analysis_type')}")
    print(f"  - Next node: {result2.get('next_node')}")
    print(f"  - Need research: {result2.get('need_research')}")
    print(f"  - CNN result: {result2.get('cnn_result')}")
    print(f"  - VLM result: {result2.get('vlm_result')}")
    print(f"  - Search results count: {len(result2.get('search_results', []))}")
    if result2.get('search_results'):
        print(f"  - First search result: {result2['search_results'][0]}")
    print(f"  - Final response: {result2.get('final_response')[:200] + '...' if result2.get('final_response') else 'N/A'}")
    
    print("\n" + "=" * 80)
    print("FULL RESULT (Phase 2B):")
    print("=" * 80)
    print(json.dumps(result2, indent=2, default=str))
    
except Exception as e:
    print(f"ERROR in Phase 2B: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[OK] Phase 2 test completed successfully")

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import json

class ValidationAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def validate_result(self, state):
        """
        Validates the fusion result and assigns a confidence score.
        """
        fusion_result = state.get("fusion_result", "")
        vision_data = state.get("vision_insights", "")
        text_data = state.get("text_insights", "")
        
        prompt = f"""
        You are a Validation Agent. Your job is to assess the quality and coherence of the document analysis.
        
        Fusion Result: {fusion_result[:1000]}...
        Vision Insights: {vision_data[:500]}...
        Text Insights: {text_data[:500]}...
        
        Task:
        1. Rate your confidence in the Fusion Result on a scale of 0.0 to 1.0.
        2. Provide a brief explanation for the score.
        3. Return ONLY a JSON object with keys: "confidence_score" (float) and "validation_notes" (string).
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            # Basic cleanup if markdown backticks are used
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
                
            data = json.loads(content)
            return {
                "jit_confidence_score": data.get("confidence_score", 0.5),
                "validation_notes": data.get("validation_notes", "No notes provided."),
                "final_output": {
                    "summary": fusion_result,
                    "confidence": data.get("confidence_score", 0.5),
                    "notes": data.get("validation_notes", "")
                }
            }
        except Exception as e:
            print(f"Validation Agent Error: {e}")
            return {
                "jit_confidence_score": 0.0, 
                "validation_notes": f"Validation Error: {e}",
                 "final_output": {"error": str(e)}
            }

from typing import TypedDict, Annotated, List, Any
import operator
from PIL import Image

class AgentState(TypedDict):
    # Inputs
    file_path: str
    images: List[Any] # PIL Images
    
    # Intermediate Processing
    detected_layout: List[dict] # From CV Service
    ocr_text: str # From OCR Service
    
    # Agent Outputs
    vision_insights: str # From Vision Agent
    text_insights: str # From Text Agent
    fusion_result: str # From Fusion Agent
    
    # Validation
    jit_confidence_score: float # 0.0 to 1.0
    validation_notes: str
    
    # Final Output
    final_output: dict

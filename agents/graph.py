from langgraph.graph import StateGraph, END
from app.agents.state import AgentState
from app.agents.vision_agent import VisionAgent
from app.agents.text_agent import TextAgent
from app.agents.fusion_agent import FusionAgent
from app.agents.validation_agent import ValidationAgent

def build_graph():
    # Initialize Agents
    vision_agent = VisionAgent()
    text_agent = TextAgent()
    fusion_agent = FusionAgent()
    validation_agent = ValidationAgent()
    
    # Define Workflow
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("vision_node", vision_agent.process_visuals)
    workflow.add_node("text_node", text_agent.process_text)
    workflow.add_node("fusion_node", fusion_agent.fuse_information)
    workflow.add_node("validation_node", validation_agent.validate_result)
    
    # Define Edges
    # We run Vision and Text in parallel (conceptually in graph, but here sequentially or branched)
    # LangGraph allows branching.
    
    workflow.set_entry_point("vision_node")
    workflow.add_edge("vision_node", "text_node") # For simplicity of linear execution in this prototype
    workflow.add_edge("text_node", "fusion_node")
    workflow.add_edge("fusion_node", "validation_node")
    workflow.add_edge("validation_node", END)
    
    return workflow.compile()
    
graph = build_graph()

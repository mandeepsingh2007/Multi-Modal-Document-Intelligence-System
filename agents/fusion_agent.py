from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

class FusionAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def fuse_information(self, state):
        """
        Combines visual and textual insights.
        """
        vision_data = state.get("vision_insights", "")
        text_data = state.get("text_insights", "")
        
        prompt = f"""
        You are a Fusion Agent. specific task is to merge the information from the Visual analysis and Text analysis of a document.
        
        Visual Insights:
        {vision_data}
        
        Text Insights:
        {text_data}
        
        Provide a consolidated summary of the document, resolving any conflicts if present. 
        Structure your response clearly.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return {"fusion_result": response.content}
        except Exception as e:
            return {"fusion_result": f"Error in Fusion Agent: {e}"}

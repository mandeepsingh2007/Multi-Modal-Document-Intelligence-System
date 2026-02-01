from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from app.core.config import settings

class TextAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def process_text(self, state):
        """
        Analyzes the OCR text to extract key information and summary.
        """
        text = state.get("ocr_text", "")
        if not text:
            return {"text_insights": "No text extracted."}

        system_prompt = "You are an expert document analyst. Summarize the following text and extract key entities."
        user_message = HumanMessage(content=text[:100000]) # Increased context limit for full papers

        try:
            response = self.llm.invoke([SystemMessage(content=system_prompt), user_message])
            return {"text_insights": response.content}
        except Exception as e:
            return {"text_insights": f"Error in Text Agent: {e}"}

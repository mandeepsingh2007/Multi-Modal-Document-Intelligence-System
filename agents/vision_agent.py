from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from app.core.config import settings
import base64
import io

class VisionAgent:
    def __init__(self):
        # We prefer a model with vision capabilities.
        # Ensure OPENAI_API_KEY is set in environment.
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def image_to_base64(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def process_visuals(self, state):
        """
        Analyzes the images to extract insights about tables, charts, and layout.
        """
        images = state.get("images", [])
        if not images:
            return {"vision_insights": "No images provided."}

        # For efficiency, we might only process the first page or specific crops in a real system.
        # Here we process the first page as a sample.
        first_page = images[0]
        img_b64 = self.image_to_base64(first_page)

        message = HumanMessage(
            content=[
                {"type": "text", "text": "Analyze this document image. Describe any tables, charts, or diagrams you see in detail. Ignore standard text if possible, focus on visual elements and layout structure."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        )

        try:
            response = self.llm.invoke([message])
            return {"vision_insights": response.content}
        except Exception as e:
            return {"vision_insights": f"Error in Vision Agent: {e}"}

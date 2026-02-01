import easyocr
import numpy as np
from PIL import Image

class OCRService:
    def __init__(self, lang_list: list[str] = ['en']):
        print("Initializing OCR Engine (this may take a moment)...")
        self.reader = easyocr.Reader(lang_list)

    def extract_text(self, image: Image.Image) -> str:
        """
        Extract detailed text from an entire image or a crop.
        """
        img_array = np.array(image)
        # paragraph=True helps combine lines into coherent blocks, better for detailed extraction
        result = self.reader.readtext(img_array, detail=0, paragraph=True)
        return "\n\n".join(result)

    def extract_text_with_layout(self, image: Image.Image) -> list[dict]:
        """
        Returns list of {'text': str, 'bbox': [x1, y1, x2, y2]}
        """
        img_array = np.array(image)
        result = self.reader.readtext(img_array)
        
        structured_text = []
        for (bbox, text, prob) in result:
            # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            x1 = min([p[0] for p in bbox])
            y1 = min([p[1] for p in bbox])
            x2 = max([p[0] for p in bbox])
            y2 = max([p[1] for p in bbox])
            
            structured_text.append({
                "text": text,
                "bbox": [x1, y1, x2, y2],
                "confidence": prob
            })
            
        return structured_text

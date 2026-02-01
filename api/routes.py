from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.ingestion_service import IngestionService
from app.services.cv_service import CVService
from app.services.ocr_service import OCRService
from app.rag.embedding import EmbeddingService
from app.rag.vector_store import VectorStore
from app.agents.graph import graph
import shutil
import os

router = APIRouter()

# Initialize services GLOBALLY so they persist in memory
# This is crucial for QdrantClient(":memory:") to work across requests
ingestion_service = IngestionService()
cv_service = CVService() 
ocr_service = OCRService()
embed_service = EmbeddingService()
vector_store = VectorStore()

@router.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    Upload a PDF document and get a multi-modal analysis.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        # Read file
        contents = await file.read()
        
        # 1. Ingestion
        images = ingestion_service.convert_pdf_to_images(contents)
        if not images:
             raise HTTPException(status_code=400, detail="Could not process PDF.")
             
        # Limit to first page for prototype speed
        focus_image = images[0]
        
        # 2. CV Analysis
        layout = cv_service.analyze_layout(focus_image)
        
        # 3. OCR Analysis
        # Try direct PDF extraction first (Digital PDF)
        pdf_text = ingestion_service.extract_text_from_pdf(contents)
        
        if len(pdf_text) > 100:
            ocr_text = pdf_text
            print("DEBUG: Using digital PDF text extraction (Hybrid Mode).")
        else:
            # Fallback to OCR (Scanned PDF) - Process ALL pages
            print("DEBUG: Digital extraction failed. Running OCR on ALL pages...")
            ocr_chunks = []
            for i, img in enumerate(images):
                print(f"DEBUG: OCR Processing page {i+1}/{len(images)}")
                page_text = ocr_service.extract_text(img)
                ocr_chunks.append(page_text)
            ocr_text = "\n\n".join(ocr_chunks)
        
        # 4. Agentic Workflow
        initial_state = {
            "file_path": file.filename,
            "images": images, # Agents can now access all images
            "detected_layout": layout, # Layout still from first page for quick visual summary
            "ocr_text": ocr_text,
            "vision_insights": "",
            "text_insights": "",
            "fusion_result": "",
            "jit_confidence_score": 0.0,
            "validation_notes": "",
            "final_output": {}
        }
        
        # Fix: Ensure images are passed as list even if one
        if not isinstance(images, list): images = [images]
        result = graph.invoke(initial_state)
        
        # 5. RAG Indexing
        # Index the SUMMARY
        final_summary = result.get("final_output", {}).get("summary", "")
        if final_summary:
            print(f"DEBUG: Indexing summary of length {len(final_summary)}")
            embedding = embed_service.get_embedding(final_summary)
            vector_store.add_document(
                text=final_summary,
                metadata={"filename": file.filename, "type": "summary"},
                embedding=embedding
            )

        # Index the RAW OCR TEXT (Chunking would be better for production, but this solves the immediate missing detail issue)
        raw_text = initial_state.get("ocr_text", "")
        if raw_text:
            print(f"DEBUG: Indexing raw text of length {len(raw_text)}")
            # Split roughly if too large (naive chunking for prototype)
            chunk_size = 4000
            for i in range(0, len(raw_text), chunk_size):
                chunk = raw_text[i:i+chunk_size]
                embedding = embed_service.get_embedding(chunk)
                vector_store.add_document(
                    text=chunk,
                    metadata={"filename": file.filename, "type": "raw_text", "chunk_index": i},
                    embedding=embedding
                )
            print("DEBUG: Raw text indexed successfully.")
            
        return result["final_output"]
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query")
async def query_document(query_request: dict):
    """
    Query the indexed documents.
    Payload: {"query": "string"}
    """
    query_text = query_request.get("query")
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text required")
        
    try:
        print(f"DEBUG: Querying for '{query_text}'")
        # 1. Embed query
        query_vec = embed_service.get_embedding(query_text)
        
        # 2. Search
        # Increase limit to capture more context (e.g. author lists spanning multiple lines)
        results = vector_store.search(query_vec, limit=10)
        print(f"DEBUG: Found {len(results)} results")
        
        # 3. Generate Answer using LLM
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        if not results:
             return {"results": [{"text": "I couldn't find any relevant information in the document.", "score": 0.0}]}

        # Combine context from top results
        context_text = "\n\n".join([res.payload.get("text", "") for res in results])
        
        # Initialize LLM (lightweight instantiation)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Create prompt
        system_prompt = "You are a helpful assistant. Answer the user's question based ONLY on the provided context. Be concise and direct."
        user_message = f"Context:\n{context_text}\n\nQuestion: {query_text}"
        
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_message)])
        
        # Return generated answer
        return {"results": [{"text": response.content, "score": 1.0}]}
        
    except Exception as e:
        import traceback
        traceback.print_exc() # <--- This will print the error to the terminal
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
def health_check():
    return {"status": "ok"}

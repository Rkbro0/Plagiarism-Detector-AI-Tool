from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from models import Document
from plagiarism_detector import PlagiarismDetector

def create_app(detector: PlagiarismDetector, api_key: str):
    app = FastAPI(
        title="Plagiarism Detection API",
        description="An API for detecting plagiarism in text documents using TF-IDF and cosine similarity.",
        version="1.0.0",
    )

    api_key_header = APIKeyHeader(name="X-API-Key")

    def get_api_key(api_key_header: str = Depends(api_key_header)):
        if api_key_header != api_key:
            raise HTTPException(status_code=403, detail="Could not validate credentials")
        return api_key_header

    @app.post("/add_document", response_model=dict, dependencies=[Depends(get_api_key)])
    async def add_document(document: Document):
        document_id = detector.add_document(document.text)
        return {"message": "Document added successfully", "document_id": document_id}

    @app.post("/check_plagiarism", response_model=dict, dependencies=[Depends(get_api_key)])
    async def check_plagiarism(document: Document):
        try:
            result = detector.check_plagiarism(document.text)
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/document/{document_id}", response_model=dict, dependencies=[Depends(get_api_key)])
    async def get_document(document_id: int):
        try:
            text = detector.get_document(document_id)
            return {"document_id": document_id, "text": text}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    return app

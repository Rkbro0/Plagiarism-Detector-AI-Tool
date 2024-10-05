import uvicorn
from plagiarism_detector import PlagiarismDetector
from api import create_app
from config import API_KEY

detector = PlagiarismDetector()
app = create_app(detector, API_KEY)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.DiseaseIdentify import diseaseIdentify
from routes.TextIdentify import textIdentify

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(diseaseIdentify,prefix="/disease-identify")
app.include_router(textIdentify,prefix="/text-identify")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
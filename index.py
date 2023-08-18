from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from routes.test import test
# from routes.DiseaseIdentify import diseaseIdentify
from routes.TextIdentify import textIdentify

app = FastAPI()

origins = [
    "http://127.0.0.1:8000",
    "http://localhost:19006"
]

# You can customize other CORS settings as well, like methods, headers, etc.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],   # You can specify specific HTTP methods here
    allow_headers=["*"],   # You can specify specific HTTP headers here
)

# app.include_router(test,prefix="/test")
# app.include_router(diseaseIdentify,prefix="/disease-identify")
app.include_router(textIdentify,prefix="/text-identify")

# Run the FastAPI server using Uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
from fastapi import FastAPI
# from routes.test import test
# from routes.DiseaseIdentify import diseaseIdentify
from routes.TextIdentify import textIdentify

app = FastAPI()

# app.include_router(test,prefix="/test")
# app.include_router(diseaseIdentify,prefix="/disease-identify")
app.include_router(textIdentify,prefix="/text-identify")
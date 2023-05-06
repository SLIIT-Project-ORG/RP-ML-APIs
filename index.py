from fastapi import FastAPI
from routes.test import test
app = FastAPI()
app.include_router(test)
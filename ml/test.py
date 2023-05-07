from fastapi import APIRouter

test = APIRouter()

@test.get("/hello")
async def sayHello():
    return "Hello"
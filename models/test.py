from pydantic import BaseModel

class Test(BaseModel):
    name:str
    description:str
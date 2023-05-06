from fastapi import APIRouter

from models.test import Test
from config.db import conn
from schemas.test import testEntity,testsEntity

test = APIRouter()

@test.get('/')
async def find_all_test():
    print(conn.get_database().test.find())
    print(conn.get_database().test.find())
    return testsEntity(conn.get_database().test.find())


@test.post('/')
async def create_test( test:Test):
    conn.get_database().test.insert_one(dict(test))
    
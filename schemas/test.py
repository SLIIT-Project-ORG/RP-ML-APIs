def testEntity(item) -> dict:
    return {
        "id":str(item["_id"]),
        "name":item["name"],
        "description":item["description"]
    }
    
def testsEntity(entity) -> list:
    return [testEntity(item) for item in entity]
    
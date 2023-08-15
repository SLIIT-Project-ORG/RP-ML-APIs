from pydantic import BaseModel,FilePath


class DiseaseIdentify(BaseModel):
    audio_file: bytes

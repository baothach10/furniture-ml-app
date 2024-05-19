
from fastapi import FastAPI, UploadFile, File, status, Form
from app.utils.handle_file import save_to_FS
from fastapi.middleware.cors import CORSMiddleware
from .constants.config import settings
from fastapi.staticfiles import StaticFiles
import os

PREFIX = f"/api/{settings.API_VERSION}"

app = FastAPI(
    openapi_url=f"{PREFIX}/openapi.json",
    docs_url=f"{PREFIX}/docs",
    redoc_url=f"{PREFIX}/redoc",
)
app.mount(
    "/static",
    StaticFiles(
        directory=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        + "/static",
        html=False,
    ),
    name="static",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pydantic import BaseModel
import json

class CreateImage(BaseModel):
    file_name: str

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value
    
@app.post(
    "/create",
    status_code=status.HTTP_201_CREATED,
)
async def add_image(
    file: UploadFile,
):
    if file:
        file_content = await file.read()
        save_to_FS("image",  file.filename, "jpg", file_content)

    return 'Success'
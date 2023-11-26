from fastapi import FastAPI, File, Depends, UploadFile
from fastapi.responses import FileResponse
import uvicorn
import tempfile
from api_handler import analyse_zip
import os
import subprocess
import asyncio

app = FastAPI()

@app.post("/analyse_chat/")
async def post_endpoint(file: bytes = File()):
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(file)
        print(os.path.exists(tmp.name))
        asyncio.create_subprocess_shell(f'python python analyse_chat.py {tmp.name}')
        response = FileResponse(path=tmp.name, filename=tmp.name)
        return response



if __name__ == "__main__":
    uvicorn.run("main:app")

from fastapi import Request, FastAPI
import uvicorn

app = FastAPI()


@app.get("/")
async def get_response(request: Request) -> list:
    response = dummy()
    return response


if __name__ == "__main__":
    uvicorn.run("main:app")

"""Main entrypoint for the app."""

import json
import logging
import os
import sys
from importlib import import_module
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from llama_cpp import Llama

sys.path.append(str(Path(__file__).parent.parent))
import resources as res
from schemas import WSMessage

DEFAULT_PORT = 8123

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

logging.basicConfig(level=logging.INFO)

config_module = (
    os.getenv("CONFIGURATION") if os.getenv("CONFIGURATION") is not None else "default"
)
logging.info(f"Configuration: {config_module}")
conf = import_module(f"configuration.{config_module}")


async def send(ws, msg: str, type: str):
    message = WSMessage(sender="bot", message=msg, type=type)
    await ws.send_json(message.dict())


@app.on_event("startup")
async def startup_event():
    global llm

    llm = Llama(
        model_path=conf.LLM_PATH,
        device=conf.DEVICE,
        n_gpu_layers=conf.GPU_LAYERS,
        n_threads=conf.N_THREADS,
        n_ctx=conf.CONTEXT_TOKENS,
    )
    logging.info("Server started")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/img/favicon.ico")


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "res": res, "conf": conf}
    )


@app.get("/inference.js")
async def get(request: Request):  # noqa: F811
    return templates.TemplateResponse(
        "inference.js",
        {"request": request, "wsurl": os.getenv("WSURL", ""), "res": res, "conf": conf},
    )


@app.websocket("/inference")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await send(
        websocket, "I'm ragger duck! Ask me question about scikit-learn!", "info"
    )

    while True:
        try:
            response_complete = ""
            start_type = ""

            received_text = await websocket.receive_text()
            payload = json.loads(received_text)

            prompt = payload["query"]
            start_type = "start"

            await send(websocket, "Analyzing prompt...", "info")
            stream = llm(
                prompt,
                echo=False,
                stream=True,
                max_tokens=conf.MAX_RESPONSE_TOKENS,
                temperature=conf.TEMPERATURE,
            )
            for i in stream:
                response_text = i.get("choices", [])[0].get("text", "")
                answer_type = start_type if response_complete == "" else "stream"
                response_complete += response_text
                await send(websocket, response_text, answer_type)
            await send(websocket, response_complete, start_type)

            await send(websocket, "", "end")
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            await send(websocket, "Sorry, something went wrong. Try again.", "error")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=DEFAULT_PORT)

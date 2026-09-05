import os
import gradio as gr
from src.backend.main import app as fastapi_app

# Minimal Gradio shell — HF Spaces needs this to recognise the file
with gr.Blocks() as demo:
    gr.Markdown("NyayBot API")  # placeholder, never seen by users

# Mount Gradio onto FastAPI — returns the combined ASGI app
# FastAPI serves the React frontend at / and API at /query, /health etc.
# Gradio lives at /gradio (unused but satisfies the SDK requirement)
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    # Hugging Face Spaces routes external traffic to port 7860.
    # On ZeroGPU Spaces, HF sets PORT=7861 for an internal worker/proxy which causes
    # an [Errno 98] address already in use error if uvicorn tries to bind to it.
    # We bind to 7860 unless an explicit non-conflicting PORT is specified.
    env_port = os.environ.get("PORT")
    port = int(env_port) if (env_port and env_port != "7861") else 7860
    uvicorn.run(app, host="0.0.0.0", port=port)


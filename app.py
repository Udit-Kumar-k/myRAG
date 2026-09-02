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
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

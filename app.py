import os
import gradio as gr

try:
    import spaces

    @spaces.GPU
    def zero_gpu_keepalive(x: str = "") -> str:
        """ZeroGPU function to satisfy HF ZeroGPU supervisor."""
        return "ZeroGPU active"
except (ImportError, AttributeError):
    def zero_gpu_keepalive(x: str = "") -> str:
        return "CPU active"

from src.backend.main import app as fastapi_app

# Minimal Gradio interface — satisfies ZeroGPU runner and Gradio SDK requirements
demo = gr.Interface(
    fn=zero_gpu_keepalive,
    inputs=gr.Textbox(label="Status Input", value="status"),
    outputs=gr.Textbox(label="Status Output"),
    title="NyayBot API"
)

# Mount Gradio onto FastAPI — returns the combined ASGI app
# FastAPI serves the React frontend at / and API at /query, /health etc.
# Gradio lives at /gradio
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



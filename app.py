import os
import gradio as gr

try:
    import spaces

    @spaces.GPU(duration=60)
    def zero_gpu_keepalive(text: str = "") -> str:
        """ZeroGPU function to satisfy HF ZeroGPU supervisor."""
        return f"ZeroGPU active: {text}"
except Exception as e:
    def zero_gpu_keepalive(text: str = "") -> str:
        return f"CPU active: {text}"

from src.backend.main import app as fastapi_app

# Minimal Gradio interface with queueing enabled for ZeroGPU compatibility
with gr.Blocks() as demo:
    gr.Markdown("## NyayBot Legal Assistant API")
    inp = gr.Textbox(label="Status Input", value="ping")
    out = gr.Textbox(label="Status Output")
    btn = gr.Button("Check Status")
    btn.click(fn=zero_gpu_keepalive, inputs=inp, outputs=out)

# Crucial for ZeroGPU: enable queueing so ZeroGPU supervisor can hook in
demo.queue()

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



# hf2ollama

A small Flask app that downloads GGUF models from Hugging Face and runs them
locally with `llama-cpp-python`. Ollama is not required.

## How it works

1. Enter a Hugging Face repository ID containing GGUF weights.
2. Select a quantized `.gguf` file and download it into `./models`.
3. Open the Chat tab and select any downloaded GGUF model.
4. The selected model is loaded directly through llama.cpp and used for chat.
5. Optionally, use the Create Ollama Model tab to register a downloaded GGUF
   with custom system instructions and temperature. The app generates and
   removes the temporary Modelfile automatically.

Only one model is kept in memory at a time. Selecting another model releases
the previous model before loading the new one.

## Requirements

- Python 3.9+
- A C/C++ toolchain if a compatible `llama-cpp-python` wheel is unavailable
- Enough RAM or VRAM for the selected GGUF quantization

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`llama-cpp-python` is a native package. For CUDA, Vulkan, Metal, and other
accelerated builds, follow its platform-specific installation documentation:
<https://github.com/abetlen/llama-cpp-python#installation>

## Run

```powershell
python app.py
```

Open <http://localhost:5050>.

## Runtime configuration

| Variable | Default | Purpose |
|---|---:|---|
| `LLAMA_N_CTX` | `4096` | Context window used when loading a model |
| `LLAMA_N_GPU_LAYERS` | `-1` | Number of layers to offload to the GPU |
| `HF_TOKEN` | unset | Access token for gated or private Hugging Face repositories |

Set `LLAMA_N_GPU_LAYERS=0` to force CPU inference.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/list-files` | List GGUF files in a Hugging Face repository |
| `POST` | `/api/pull` | Start a GGUF download job |
| `GET` | `/api/status/<job_id>` | Read download progress and status |
| `GET` | `/api/models` | List downloaded GGUF models |
| `POST` | `/api/chat` | Chat directly with a downloaded GGUF model |
| `POST` | `/api/ollama/create` | Create an Ollama model from a downloaded GGUF |

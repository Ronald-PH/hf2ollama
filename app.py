"""
hf2ollama - Download GGUF models from Hugging Face and run them locally.

Requires:
  - pip install -r requirements.txt

Run:
  python app.py
Then open http://localhost:5050
"""
import gc
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

import requests
from flask import Flask, jsonify, render_template, request
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import HfHubHTTPError, build_hf_headers

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# In-memory job store: {job_id: {status, stage, progress, log: [...], error}}
JOBS = {}
JOBS_LOCK = threading.Lock()

hf_api = HfApi()
MODEL_LOCK = threading.Lock()
LOADED_MODEL = None
LOADED_MODEL_PATH = None


def _log(job_id, message, stage=None, progress=None):
    with JOBS_LOCK:
        job = JOBS[job_id]
        job["log"].append(message)
        if stage:
            job["stage"] = stage
        if progress is not None:
            job["progress"] = progress


def _update_download_progress(job_id, downloaded, total, progress=None):
    with JOBS_LOCK:
        job = JOBS[job_id]
        job["bytes_downloaded"] = downloaded
        job["bytes_total"] = total
        job["download_pct"] = round((downloaded / total) * 100, 1) if total else None
        if progress is not None:
            job["progress"] = progress


def _fail(job_id, message):
    with JOBS_LOCK:
        job = JOBS[job_id]
        job["status"] = "error"
        job["error"] = message
        job["log"].append(f"ERROR: {message}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models")
def models():
    """List complete GGUF files downloaded into the local models directory."""
    installed = []
    for root, _, filenames in os.walk(MODELS_DIR):
        for filename in filenames:
            if not filename.lower().endswith(".gguf"):
                continue
            full_path = os.path.join(root, filename)
            relative_path = os.path.relpath(full_path, MODELS_DIR).replace(os.sep, "/")
            stat = os.stat(full_path)
            installed.append({
                "name": relative_path,
                "size": stat.st_size,
                "modified_at": stat.st_mtime,
            })
    installed.sort(key=lambda item: item["name"].lower())
    return jsonify({"models": installed})


def _resolve_model_path(model_name):
    """Resolve a client model name while preventing paths outside MODELS_DIR."""
    candidate = os.path.realpath(os.path.join(MODELS_DIR, model_name.replace("/", os.sep)))
    models_root = os.path.realpath(MODELS_DIR)
    if os.path.commonpath([candidate, models_root]) != models_root:
        return None
    if not candidate.lower().endswith(".gguf") or not os.path.isfile(candidate):
        return None
    return candidate


def _sanitize_ollama_name(name):
    """Normalize a user-entered name to Ollama's supported name characters."""
    name = re.sub(r"[^a-zA-Z0-9._:-]+", "-", name.strip()).strip("-")
    return name.lower()


def _load_model(model_path):
    """Load one GGUF at a time, releasing the previous model before switching."""
    global LOADED_MODEL, LOADED_MODEL_PATH
    if LOADED_MODEL is not None and LOADED_MODEL_PATH == model_path:
        return LOADED_MODEL

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is not installed. Run: pip install -r requirements.txt"
        ) from exc

    LOADED_MODEL = None
    LOADED_MODEL_PATH = None
    gc.collect()

    LOADED_MODEL = Llama(
        model_path=model_path,
        n_ctx=int(os.environ.get("LLAMA_N_CTX", "4096")),
        n_gpu_layers=int(os.environ.get("LLAMA_N_GPU_LAYERS", "-1")),
        verbose=False,
    )
    LOADED_MODEL_PATH = model_path
    return LOADED_MODEL


@app.route("/api/ollama/create", methods=["POST"])
def create_ollama_model():
    """Create an Ollama model from a downloaded GGUF in a background job."""
    data = request.get_json(silent=True) or {}
    gguf_name = str(data.get("gguf") or "").strip()
    model_name = _sanitize_ollama_name(str(data.get("model_name") or ""))
    system_prompt = str(data.get("system_prompt") or "").strip()
    temperature = data.get("temperature")

    model_path = _resolve_model_path(gguf_name)
    if model_path is None:
        return jsonify({"error": f"GGUF model '{gguf_name}' was not found."}), 404
    if not model_name:
        return jsonify({"error": "A valid Ollama model name is required."}), 400
    if len(system_prompt) > 50000:
        return jsonify({"error": "System instructions are too long."}), 400
    if temperature not in (None, ""):
        try:
            temperature = min(max(float(temperature), 0.0), 2.0)
        except (TypeError, ValueError):
            return jsonify({"error": "Temperature must be a number from 0 to 2."}), 400
    else:
        temperature = None

    job_id = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "running",
            "stage": "queued",
            "progress": 0,
            "log": [],
            "error": None,
            "model_name": model_name,
        }

    thread = threading.Thread(
        target=_run_ollama_create_job,
        args=(job_id, model_path, model_name, system_prompt, temperature),
        daemon=True,
    )
    thread.start()
    return jsonify({"job_id": job_id, "model_name": model_name})


def _run_ollama_create_job(job_id, model_path, model_name, system_prompt, temperature):
    modelfile_path = None
    try:
        _log(job_id, "Checking for the Ollama CLI...", stage="checking", progress=5)
        if shutil.which("ollama") is None:
            _fail(job_id, "The Ollama CLI was not found on PATH. Install Ollama and restart this app.")
            return

        lines = [f'FROM "{model_path}"']
        if system_prompt:
            escaped = system_prompt.replace('"""', '\\"\\"\\"')
            lines.append(f'SYSTEM """{escaped}"""')
        if temperature is not None:
            lines.append(f"PARAMETER temperature {temperature}")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".Modelfile", prefix="hf2ollama-",
            encoding="utf-8", delete=False,
        ) as modelfile:
            modelfile.write("\n".join(lines) + "\n")
            modelfile_path = modelfile.name

        _log(job_id, "Generated a temporary Modelfile.", stage="modelfile", progress=20)
        _log(job_id, f"Creating Ollama model '{model_name}'...", stage="creating", progress=35)
        process = subprocess.Popen(
            ["ollama", "create", model_name, "-f", modelfile_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if process.stdout:
            for line in process.stdout:
                _log(job_id, line.rstrip())
        return_code = process.wait()
        if return_code != 0:
            _fail(job_id, f"ollama create exited with code {return_code}.")
            return

        _log(job_id, f"Ollama model '{model_name}' is ready.", stage="done", progress=100)
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "success"
    except Exception as exc:
        _fail(job_id, str(exc))
    finally:
        if modelfile_path:
            try:
                os.unlink(modelfile_path)
            except OSError:
                pass


@app.route("/api/chat", methods=["POST"])
def chat():
    """Run a conversation directly against a downloaded GGUF model."""
    data = request.get_json(silent=True) or {}
    model = str(data.get("model") or "").strip()
    raw_messages = data.get("messages")

    if not model:
        return jsonify({"error": "model is required"}), 400
    if not isinstance(raw_messages, list) or not raw_messages:
        return jsonify({"error": "messages must be a non-empty list"}), 400
    if len(raw_messages) > 100:
        return jsonify({"error": "conversation is too long (maximum 100 messages)"}), 400

    messages = []
    for item in raw_messages:
        if not isinstance(item, dict) or item.get("role") not in {"system", "user", "assistant"}:
            return jsonify({"error": "each message needs a valid role and content"}), 400
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            return jsonify({"error": "message content cannot be empty"}), 400
        messages.append({"role": item["role"], "content": content[:50000]})

    model_path = _resolve_model_path(model)
    if model_path is None:
        return jsonify({"error": f"GGUF model '{model}' was not found."}), 404

    try:
        temperature = min(max(float(data.get("temperature", 0.7)), 0.0), 2.0)
        max_tokens = min(max(int(data.get("max_tokens", 1024)), 1), 8192)
        with MODEL_LOCK:
            llm = _load_model(model_path)
            payload = llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        choices = payload.get("choices") or []
        reply = (choices[0].get("message") or {}).get("content") if choices else None
        if not reply:
            return jsonify({"error": "The model returned an empty response."}), 502
        return jsonify({
            "message": {"role": "assistant", "content": reply},
            "model": model,
            "done_reason": choices[0].get("finish_reason"),
            "usage": payload.get("usage"),
        })
    except (RuntimeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": f"Model inference failed: {exc}"}), 500


@app.route("/api/list-files", methods=["POST"])
def list_files():
    """List candidate GGUF files in a Hugging Face repo."""
    data = request.get_json(force=True)
    repo_id = (data.get("repo_id") or "").strip()
    if not repo_id:
        return jsonify({"error": "repo_id is required"}), 400

    try:
        info = hf_api.model_info(repo_id, files_metadata=True)
    except HfHubHTTPError as e:
        return jsonify({"error": f"Could not find repo '{repo_id}': {e}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    gguf_files = []
    for sibling in info.siblings:
        if sibling.rfilename.lower().endswith(".gguf"):
            size = getattr(sibling, "size", None)
            gguf_files.append({
                "filename": sibling.rfilename,
                "size_mb": round(size / (1024 * 1024), 1) if size else None,
            })

    gguf_files.sort(key=lambda f: f["filename"])

    return jsonify({
        "repo_id": repo_id,
        "gguf_files": gguf_files,
        "has_gguf": len(gguf_files) > 0,
    })


@app.route("/api/pull", methods=["POST"])
def pull():
    """Kick off a background job that downloads a GGUF from Hugging Face."""
    data = request.get_json(force=True)
    repo_id = (data.get("repo_id") or "").strip()
    filename = (data.get("filename") or "").strip()
    if not repo_id or not filename:
        return jsonify({"error": "repo_id and filename are required"}), 400

    job_id = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "running",
            "stage": "queued",
            "progress": 0,
            "log": [],
            "error": None,
            "bytes_downloaded": 0,
            "bytes_total": None,
            "download_pct": None,
        }

    thread = threading.Thread(
        target=_run_pull_job,
        args=(job_id, repo_id, filename),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


def _download_with_progress(job_id, repo_id, filename):
    """Stream a file from the HF Hub to MODELS_DIR, updating job progress
    as bytes arrive. Returns the local file path."""
    url = hf_hub_url(repo_id=repo_id, filename=filename)
    token = os.environ.get("HF_TOKEN")
    headers = build_hf_headers(token=token)

    dest_path = os.path.join(MODELS_DIR, os.path.basename(filename))
    os.makedirs(os.path.dirname(dest_path) or MODELS_DIR, exist_ok=True)
    tmp_path = dest_path + ".part"

    with requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=30) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length") or 0) or None
        downloaded = 0
        last_update = 0.0
        chunk_size = 8 * 1024 * 1024  # 8 MB chunks

        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_update > 0.3 or (total and downloaded >= total):
                    # Map raw download percent (0-100) onto the overall
                    # job progress range reserved for this stage (15-60).
                    overall = 15
                    if total:
                        overall = 15 + (downloaded / total) * 45
                    _update_download_progress(job_id, downloaded, total, progress=round(overall, 1))
                    last_update = now

    os.replace(tmp_path, dest_path)
    return dest_path


def _run_pull_job(job_id, repo_id, filename):
    try:
        # Download the GGUF file from Hugging Face, streaming so we can
        #    report real byte-level progress back to the UI.
        _log(job_id, f"Downloading {filename} from {repo_id}...", stage="downloading", progress=5)
        local_path = _download_with_progress(job_id, repo_id, filename)
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        _log(job_id, f"Downloaded {size_mb:.1f} MB to {local_path}", stage="done", progress=100)
        _log(job_id, "The GGUF model is ready in the Chat tab.")
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "success"

    except HfHubHTTPError as e:
        _fail(job_id, f"Hugging Face download failed: {e}")
    except Exception as e:
        _fail(job_id, str(e))


@app.route("/api/status/<job_id>")
def status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "unknown job_id"}), 404
        return jsonify(job)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)

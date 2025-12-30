import modal
import os

# --- Configuration ---
MODEL_DIR = "/model_cache"
QWEN_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
# ---------------------

stub = modal.App(
    name="model-downloader",
    image=modal.Image.debian_slim().pip_install(
        "transformers",
        "sentence-transformers",
        "torch",
        "accelerate",
        "bitsandbytes"
    ),
)

# This is your persistent network drive on Modal.
# THE FIX IS HERE:
# NEW (Just look up the volume you created in Step 1)
volume = modal.Volume.from_name("model-cache-vol")
# END OF FIX

# This is the function you will run at the cafe.
# We mount the volume at /model_cache
@stub.function(
    volumes={MODEL_DIR: volume},
    timeout=5400,  # 30 minutes, just in case
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def download_models():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer

    # 1. Download the Qwen LLM
    print(f"--- Downloading Qwen model: {QWEN_MODEL_NAME} ---")
    AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
        cache_dir=MODEL_DIR,
        load_in_4bit=True, # We add this to match the runner
    )
    AutoTokenizer.from_pretrained(
        QWEN_MODEL_NAME,
        cache_dir=MODEL_DIR,
    )
    print("--- Qwen Download Complete ---")

    # 2. Download the Embedding Model
    print(f"--- Downloading Embedding model: {EMBEDDING_MODEL_NAME} ---")
    SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        cache_folder=MODEL_DIR,
    )
    print("--- Embedding Model Download Complete ---")

    # 3. Commit the changes to the volume
    print("Committing files to volume...")
    volume.commit()
    print("All models downloaded and saved to volume.")
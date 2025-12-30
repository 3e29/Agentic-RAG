import modal
import os

# --- 1. IMPORT THE NEW CONFIG OBJECT ---
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import torch

# --- Configuration ---
MODEL_DIR = "/model_cache"
QWEN_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
# ---------------------

image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers",
        "torch",
        "accelerate",
        "bitsandbytes",
        "sentence_transformers",
        "fastapi"  # <--- THIS IS THE FIX
    )
)

# Use 'app' as the variable name
app = modal.App(name="qwen2.5-14b-runner", image=image)
volume = modal.Volume.from_name("model-cache-vol")

@app.cls(
    gpu="A10G",
    scaledown_window=300, # Use new name
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
class QwenModel:
    
    def __enter__(self):
        import os
        HF_TOKEN = os.environ["HF_TOKEN"]
        
        # --- DEFINE THE 4-BIT QUANTIZATION CONFIG ---
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)

        print("--- Loading Model from Volume ---")
        
        # --- A) Load Qwen LLM ---
        self.tokenizer = AutoTokenizer.from_pretrained(
            QWEN_MODEL_NAME,
            cache_dir=MODEL_DIR,
            token=HF_TOKEN
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb_config, # Use new config
            cache_dir=MODEL_DIR,
            token=HF_TOKEN
        )
        
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        print("--- Qwen LLM Loaded ---")

        # --- B) Load Embedding Model ---
        print("--- Loading Embedding Model from Volume ---")
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            cache_folder=MODEL_DIR,
            device="cuda"
        )
        print("--- Embedding Model Loaded ---")
        
        return self

    # Use '@modal.fastapi_endpoint'
    @modal.fastapi_endpoint(method="POST")
    def generate(self, item: dict):
        prompt = item.get("prompt")
        if not prompt:
            return {"error": "No prompt provided"}, 400

        messages = [{"role": "user", "content": prompt}]
        prompt_formatted = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        outputs = self.pipeline(
            prompt_formatted,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
        response_text = outputs[0]["generated_text"][len(prompt_formatted):]
        
        return {"response": response_text}

    # Use '@modal.fastapi_endpoint'
    @modal.fastapi_endpoint(method="POST")
    def embed(self, item: dict):
        text = item.get("text")
        if not text:
            return {"error": "No text provided"}, 400
        
        texts = [text] if isinstance(text, str) else text
        embeddings = self.embedding_model.encode(texts)
        embeddings_list = embeddings.tolist()
        
        return {"embeddings": embeddings_list}
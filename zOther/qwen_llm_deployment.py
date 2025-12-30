"""
Modal Deployment Script for Qwen2.5-14B-Instruct LLM

This script deploys the Qwen/Qwen2.5-14B-Instruct model to Modal
for text generation, reasoning, and query analysis tasks.

Model: Qwen/Qwen2.5-14B-Instruct
- 14B parameter instruction-tuned model
- Supports long context (up to 32K tokens)
- Optimized for chat, reasoning, and instruction following
- Multilingual support including Arabic

Usage:
    modal deploy zOther/qwen_llm_deployment.py
"""

from pathlib import Path
from typing import Optional

import modal

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
MODEL_DIR = "/models"
CACHE_DIR = "/cache"

# Create Modal volumes for model caching
model_volume = modal.Volume.from_name("qwen-model-cache", create_if_missing=True)
cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# Create Modal app
app = modal.App("qwen2-5-14b-instruct")

# Define the image with all dependencies
llm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers==4.48.1",
        "torch==2.5.1",
        "accelerate==1.2.1",
        "huggingface_hub[hf_transfer]",
        "fastapi[standard]",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": CACHE_DIR,
        "TRANSFORMERS_CACHE": CACHE_DIR,
    })
)


def download_model_func():
    """
    Download the Qwen2.5-14B-Instruct model from Hugging Face.
    This runs once during image build to cache the model.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    print(f"Downloading model: {MODEL_NAME}")
    
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_DIR,
        trust_remote_code=True,
    )
    
    # Download model (fp16 to save space)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_DIR,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    print(f"Model downloaded successfully to {MODEL_DIR}")
    print(f"Model config: {model.config}")


# Build the inference image with the downloaded model
inference_image = (
    llm_image
    .run_function(
        download_model_func,
        volumes={
            MODEL_DIR: model_volume,
            CACHE_DIR: cache_volume,
        },
        timeout=3600,  # 1 hour timeout for large model download
    )
)


@app.cls(
    image=inference_image,
    gpu="A100-40GB",  # A100 GPU for faster inference, fallback to A10G
    volumes={
        MODEL_DIR: model_volume,
        CACHE_DIR: cache_volume,
    },
    min_containers=0,  # Scale to zero when not in use
    scaledown_window=300,  # 5 minutes idle timeout
    timeout=600,  # 10 minutes max execution time
)
class QwenLLM:
    """
    Qwen2.5-14B-Instruct LLM class for text generation via API.
    """
    
    @modal.enter()
    def load_model(self):
        """Load the model when the container starts."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch
        
        print(f"Loading model from {MODEL_DIR}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_DIR,
            trust_remote_code=True,
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_DIR,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Create pipeline for easier generation
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )
        
        print(f"Model loaded successfully")
        print(f"Model device: {self.model.device}")
        print(f"Model dtype: {self.model.dtype}")
    
    @modal.method()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text based on the input prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling (vs greedy decoding)
            
        Returns:
            Generated text string
        """
        # Format as chat message for instruction-tuned model
        messages = [{"role": "user", "content": prompt}]
        
        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Generate
        outputs = self.pipeline(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            return_full_text=False,  # Only return generated text
        )
        
        # Extract generated text
        generated_text = outputs[0]["generated_text"]
        
        return generated_text
    
    @modal.method()
    def generate_with_system_prompt(
        self,
        prompt: str,
        system_prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text with a system prompt.
        
        Args:
            prompt: User input prompt
            system_prompt: System instructions
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Generated text string
        """
        # Format with system message
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Generate
        outputs = self.pipeline(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            return_full_text=False,
        )
        
        # Extract generated text
        generated_text = outputs[0]["generated_text"]
        
        return generated_text


@app.function(
    image=inference_image,
    volumes={
        MODEL_DIR: model_volume,
        CACHE_DIR: cache_volume,
    },
)
def test_generation():
    """
    Test the LLM with sample prompts in English and Arabic.
    """
    llm_instance = QwenLLM()
    
    # Test prompts
    test_prompts = [
        {
            "prompt": "What is artificial intelligence?",
            "description": "English question"
        },
        {
            "prompt": "ما هو الذكاء الاصطناعي؟",
            "description": "Arabic question"
        },
        {
            "prompt": "Explain the concept of machine learning in simple terms.",
            "description": "English explanation request"
        },
    ]
    
    print("\nTesting text generation...")
    
    for i, test in enumerate(test_prompts, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test['description']}")
        print(f"{'='*60}")
        print(f"Prompt: {test['prompt']}")
        
        response = llm_instance.generate.remote(
            test['prompt'],
            max_new_tokens=200,
            temperature=0.7,
        )
        
        print(f"\nResponse:\n{response}")
    
    print("\nTest completed successfully!")


# Web endpoint class for external API calls
@app.cls(
    image=inference_image,
    gpu="A100-40GB",  # A100 GPU for fast inference
    volumes={
        MODEL_DIR: model_volume,
        CACHE_DIR: cache_volume,
    },
    min_containers=0,
    scaledown_window=300,
    timeout=600,
)
class QwenEndpoint:
    """
    Qwen2.5-14B-Instruct endpoint with persistent model loading.
    Model loads once per container and serves multiple requests.
    """
    
    @modal.enter()
    def load_model(self):
        """Load the model once when container starts."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch
        
        print(f"Loading model from {MODEL_DIR}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_DIR,
            trust_remote_code=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_DIR,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )
        
        print(f"Model loaded successfully on {self.model.device}")
    
    @modal.fastapi_endpoint(method="POST")
    def generate(self, request: dict) -> dict:
        """
        Web endpoint for text generation.
        
        Request format:
            {
                "prompt": "user prompt text",
                "system_prompt": "optional system instructions",
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9
            }
        
        Response format:
            {
                "response": "generated text"
            }
        """
        
        # Extract parameters
        prompt = request.get("prompt")
        if not prompt:
            return {"error": "No prompt provided"}
        
        system_prompt = request.get("system_prompt")
        max_new_tokens = request.get("max_new_tokens", 512)  # Reduced default for faster responses
        temperature = request.get("temperature", 0.7)
        top_p = request.get("top_p", 0.9)
        
        # Validate parameters
        if max_new_tokens > 4096:
            max_new_tokens = 4096
        if temperature < 0 or temperature > 2:
            temperature = 0.7
        if top_p < 0 or top_p > 1:
            top_p = 0.9
        
        # Generate text
        try:
            # Use do_sample based on temperature
            do_sample = temperature > 0.0
            
            # Format as chat message
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
            
            # Apply chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # Count input tokens
            prompt_tokens = len(self.tokenizer.encode(formatted_prompt))
            
            # Generate
            outputs = self.pipe(
                formatted_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                return_full_text=False,
            )
            
            response = outputs[0]["generated_text"]
            
            # Count output tokens
            completion_tokens = len(self.tokenizer.encode(response))
            total_tokens = prompt_tokens + completion_tokens
            
            return {
                "response": response,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            }
        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}


# CLI commands
@app.local_entrypoint()
def main(action: str = "test"):
    """
    Local entrypoint for testing and management.
    
    Args:
        action: Action to perform (test)
    """
    if action == "test":
        print("Running LLM generation tests...")
        test_generation.remote()
    else:
        print(f"Unknown action: {action}")
        print("Available actions: test")


"""
Deployment Instructions:
========================

1. Install Modal CLI (if not already installed):
   pip install modal

2. Setup Modal token (if not already set up):
   modal token new

3. Deploy the Qwen LLM:
   modal deploy zOther/qwen_llm_deployment.py

   This will:
   - Download the 14B model (~28GB)
   - Cache it in Modal volumes
   - Deploy the inference endpoint

4. Test the deployment:
   modal run zOther/qwen_llm_deployment.py

5. Get the web endpoint URL:
   After deployment, Modal will provide a URL like:
   https://[your-username]--qwen2-5-14b-instruct-generate.modal.run

6. Update your code to use the new endpoint:
   Update QWEN_ENDPOINT in src/utils/llm_helper.py

Example API Usage:
==================

import httpx

# Simple prompt
response = httpx.post(
    "https://[your-username]--qwen2-5-14b-instruct-generate.modal.run",
    json={"prompt": "What is machine learning?"},
    timeout=60
)
print(response.json()["response"])

# With system prompt
response = httpx.post(
    "https://[your-username]--qwen2-5-14b-instruct-generate.modal.run",
    json={
        "prompt": "Correct the spelling errors in this text: 'Wht is machene lerning?'",
        "system_prompt": "You are a helpful spelling correction assistant.",
        "temperature": 0.3,
        "max_new_tokens": 200
    },
    timeout=60
)
print(response.json()["response"])

Cost Optimization:
==================

This deployment is optimized for cost:
- min_containers=0: Scales to zero when not in use
- scaledown_window=300: Shuts down after 5 minutes of inactivity
- A10G GPU: More cost-effective than H100 for this model size
- fp16 precision: Reduces memory usage and increases speed

Performance:
============

Expected latency:
- Cold start: 30-60 seconds (model loading)
- Warm inference: 1-3 seconds per request
- Throughput: ~20-30 tokens/second on A10G

GPU Options:
============

You can change the GPU in the @app.cls decorator:
- "T4": Cheapest, slower (~10 tokens/sec)
- "A10G": Balanced cost/performance (recommended)
- "A100": Faster but more expensive (~40 tokens/sec)
- "H100": Fastest but most expensive (~60 tokens/sec)
"""


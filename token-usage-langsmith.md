Yes, to see actual token counts in LangSmith, your Modal endpoint needs to return the token usage data.

Currently your endpoint only returns:
{"response": "generated text..."}

It needs to return:
{
  "response": "generated text...",
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 45,
    "total_tokens": 195
  }
}

How to update your Modal endpoint
In your Modal deployment code, after generation, you can get token counts from the Qwen tokenizer:
# In your Modal endpoint
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

@app.function(...)
def generate(prompt: str, system_prompt: str = None, ...):
    # Build full prompt
    full_prompt = f"{system_prompt}\n{prompt}" if system_prompt else prompt
    
    # Count input tokens
    prompt_tokens = len(tokenizer.encode(full_prompt))
    
    # Generate response
    response = model.generate(...)
    
    # Count output tokens
    completion_tokens = len(tokenizer.encode(response))
    
    return {
        "response": response,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }



Then in your llm_helper.py, you'd update the code to pass this to LangSmith using langsmith.run_trees.get_current_run_tree().

Alternative: If you don't want to modify Modal, I can add approximate token estimation (character count ÷ 4) which gives a rough estimate. It won't be exact but it's better than 0. Want me to do that instead?
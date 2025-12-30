import json
import re

with open("../../data/chunks/bukhari_chunks.jsonl", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("../../data/chunks/bukhari_chunks1.jsonl", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
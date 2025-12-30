import json
import re

# Regex pattern that removes Arabic diacritics EXCEPT the shadda (ّ U+0651)
DIACRITICS_RE = re.compile(
    r'[\u0610-\u061A\u064B-\u0650\u0652-\u065F\u0670\u06D6-\u06ED]+',
    flags=re.UNICODE
)

def remove_diacritics_keep_shadda(text: str) -> str:
    """Remove Arabic diacritics but keep the shadda."""
    if not isinstance(text, str):
        return text
    return DIACRITICS_RE.sub('', text)

def recursive_clean(obj):
    """Recursively clean all strings in a JSON-like structure."""
    if isinstance(obj, dict):
        return {k: recursive_clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_clean(v) for v in obj]
    elif isinstance(obj, str):
        return remove_diacritics_keep_shadda(obj)
    else:
        return obj

def clean_json_diacritics():
    """Read the JSON, clean it, and write to a new file."""
    input_path = './muslim.json'
    output_path = './muslim.json'

    with open(input_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    cleaned_data = recursive_clean(data)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(cleaned_data, outfile, ensure_ascii=False, indent=4)

    print(f"✅ Cleaned JSON saved to: {output_path}")

if __name__ == "__main__":
    clean_json_diacritics()

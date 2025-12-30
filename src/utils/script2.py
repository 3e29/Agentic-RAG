import re

with open("./muslim.json", "r", encoding="utf-8") as f:
    data = f.read()

# Remove directionality marks
cleaned = re.sub(r'[\u200e\u200f]', '', data)

with open("./muslim.json", "w", encoding="utf-8") as f:
    f.write(cleaned)

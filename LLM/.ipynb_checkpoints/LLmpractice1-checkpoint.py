with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("the total number of characters:", len(raw_text))
print(raw_text[:100])
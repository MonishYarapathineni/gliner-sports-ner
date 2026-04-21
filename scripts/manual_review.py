# scripts/manual_review.py
import json
from pathlib import Path

test_path = Path("../data/splits/test.jsonl")
golden_path = Path("../data/splits/golden.jsonl")

examples = []
with open(test_path) as f:
    for line in f:
        examples.append(json.loads(line))

# Take first 50
to_review = examples[:50]
golden = []

print("Review each example. Press ENTER to accept, 'e' to edit, 's' to skip.\n")

for i, ex in enumerate(to_review):
    print(f"\n--- Example {i+1}/50 ---")
    print(f"TEXT: {ex['text']}\n")
    print("ENTITIES:")
    for ent in ex["entities"]:
        print(f"  [{ent['label']}] '{ex['text'][ent['start']:ent['end']]}'")
    
    choice = input("\nAccept (enter) / Edit (e) / Skip (s)? ").strip().lower()
    
    if choice == "s":
        continue
    elif choice == "e":
        print("Enter corrected JSON entities or press enter to keep:")
        raw = input().strip()
        if raw:
            try:
                ex["entities"] = json.loads(raw)
            except json.JSONDecodeError:
                print("Invalid JSON, keeping original")
    
    golden.append(ex)

with open(golden_path, "w") as f:
    for ex in golden:
        f.write(json.dumps(ex) + "\n")

print(f"\nSaved {len(golden)} golden examples to {golden_path}")
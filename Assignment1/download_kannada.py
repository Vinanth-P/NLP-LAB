from datasets import load_dataset

OUTPUT_FILE = "original_kannada.txt"
MAX_PARAGRAPHS = 1000000

dataset = load_dataset(
    "ai4bharat/IndicCorpV2",
    name="indiccorp_v2",
    split="kan_Knda",
    streaming=True
)

with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for idx, example in enumerate(dataset):

        if idx >= MAX_PARAGRAPHS:
            break

        text = example["text"].strip()

        if text:
            fout.write(text + "\n\n")

print(f"Downloaded {MAX_PARAGRAPHS} paragraphs.")
print(f"Saved to: {OUTPUT_FILE}")
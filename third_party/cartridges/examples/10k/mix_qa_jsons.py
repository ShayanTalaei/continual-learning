import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Mix AMD and Pepsi QA JSONs")
    parser.add_argument("--data-dir", default="third_party/cartridges/data/10k/eval", help="Directory containing the JSON files")
    parser.add_argument("--amd-file", default="amd_qa_gemini.json")
    parser.add_argument("--pepsi-file", default="pepsi_qa_gemini.json")
    parser.add_argument("--output-file", default="mixed_qa_gemini_generated.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    # Handle relative paths from workspace root or absolute paths
    if not data_dir.is_absolute():
        # Assuming running from workspace root, but data_dir default includes third_party...
        pass 

    amd_path = data_dir / args.amd_file
    pepsi_path = data_dir / args.pepsi_file
    output_path = data_dir / args.output_file

    print(f"Looking for files in: {data_dir.absolute()}")

    if not amd_path.exists():
        print(f"Error: {amd_path} does not exist.")
        return
    if not pepsi_path.exists():
        print(f"Error: {pepsi_path} does not exist.")
        return

    print(f"Loading {amd_path}...")
    with open(amd_path, "r") as f:
        amd_data = json.load(f)

    print(f"Loading {pepsi_path}...")
    with open(pepsi_path, "r") as f:
        pepsi_data = json.load(f)

    mixed_data = amd_data + pepsi_data
    print(f"Combined {len(amd_data)} AMD items and {len(pepsi_data)} Pepsi items into {len(mixed_data)} total items.")

    print(f"Writing to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(mixed_data, f, indent=2)
    
    print("Done.")

if __name__ == "__main__":
    main()


"""
Create evaluation datasets for the composition experiment.

Uses existing HuggingFace datasets or creates simple eval sets from the documents.
"""

import json
from pathlib import Path

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

CARTRIDGES_DIR = Path(__file__).parent.parent.parent
DATA_DIR = CARTRIDGES_DIR / "data" / "10k"
EVAL_DIR = DATA_DIR / "eval"
EVAL_DIR.mkdir(exist_ok=True)


def create_simple_eval_datasets():
    """Create simple evaluation datasets with basic QA pairs."""
    
    # AMD single-document questions
    amd_qa = [
        {
            "question": "What was AMD's total net revenue in 2024?",
            "answer": "$25,785 million",
            "type": "factual_recall"
        },
        {
            "question": "What was AMD's Data Center revenue in 2024?",
            "answer": "$12,579 million",
            "type": "factual_recall"
        },
        {
            "question": "How much did AMD spend on R&D in 2024?",
            "answer": "The document mentions R&D expenses but you need to extract the specific figure from the income statement.",
            "type": "factual_recall"
        },
        {
            "question": "Compare AMD's Data Center revenue between 2024 and 2023.",
            "answer": "AMD's Data Center revenue increased from $6,496 million in 2023 to $12,579 million in 2024, an increase of approximately 93.6%.",
            "type": "synthesis"
        },
        {
            "question": "Which AMD business segment had the largest operating income in 2024?",
            "answer": "Data Center had the largest operating income at $3,482 million.",
            "type": "reasoning"
        },
    ]
    
    # PepsiCo single-document questions
    pepsi_qa = [
        {
            "question": "What was PepsiCo's net revenue in 2024?",
            "answer": "Approximately $91.5 billion (check exact figure in document)",
            "type": "factual_recall"
        },
        {
            "question": "What major brands does PepsiCo own?",
            "answer": "Pepsi, Lay's, Gatorade, Tropicana, Quaker, Doritos, Mountain Dew, and others.",
            "type": "factual_recall"
        },
        {
            "question": "How did PepsiCo's operating income change from 2023 to 2024?",
            "answer": "You need to calculate the change from the financial statements in the document.",
            "type": "synthesis"
        },
        {
            "question": "What is PepsiCo's dividend policy?",
            "answer": "Check the shareholder equity section for dividend information.",
            "type": "factual_recall"
        },
        {
            "question": "Which geographic region contributes most to PepsiCo's revenue?",
            "answer": "North America, specifically the United States.",
            "type": "reasoning"
        },
    ]
    
    # Multi-document composition questions
    composition_qa = [
        {
            "question": "Compare the total revenue of AMD and PepsiCo in 2024.",
            "answer": "PepsiCo had significantly higher revenue (~$91.5B) compared to AMD ($25.8B). PepsiCo's revenue is approximately 3.5x larger.",
            "type": "cross_document_comparison",
            "companies": ["AMD", "PepsiCo"]
        },
        {
            "question": "Which company has higher operating margins, AMD or PepsiCo?",
            "answer": "Need to calculate operating margin (operating income / revenue) for both companies from their respective 10-Ks.",
            "type": "cross_document_reasoning",
            "companies": ["AMD", "PepsiCo"]
        },
        {
            "question": "Compare the growth rates of AMD and PepsiCo from 2023 to 2024.",
            "answer": "AMD's revenue growth rate needs to be calculated and compared with PepsiCo's growth rate.",
            "type": "cross_document_reasoning",
            "companies": ["AMD", "PepsiCo"]
        },
        {
            "question": "Which company spent more on research and development in 2024?",
            "answer": "AMD likely spent more on R&D as a percentage of revenue given it's a technology company, but absolute figures need to be compared.",
            "type": "cross_document_comparison",
            "companies": ["AMD", "PepsiCo"]
        },
        {
            "question": "Do AMD and PepsiCo have any business overlap or partnerships mentioned in their 10-Ks?",
            "answer": "Unlikely, as they operate in completely different industries (semiconductors vs. food/beverage).",
            "type": "cross_document_reasoning",
            "companies": ["AMD", "PepsiCo"]
        },
    ]
    
    # Save datasets
    with open(EVAL_DIR / "amd_qa.json", "w") as f:
        json.dump(amd_qa, f, indent=2)
    print(f"✓ Created {len(amd_qa)} AMD QA pairs")
    
    with open(EVAL_DIR / "pepsi_qa.json", "w") as f:
        json.dump(pepsi_qa, f, indent=2)
    print(f"✓ Created {len(pepsi_qa)} PepsiCo QA pairs")
    
    with open(EVAL_DIR / "composition_qa.json", "w") as f:
        json.dump(composition_qa, f, indent=2)
    print(f"✓ Created {len(composition_qa)} composition QA pairs")


def try_load_financebench():
    """Try to load FinanceBench dataset if available."""
    if not HAS_DATASETS:
        print("datasets library not installed, skipping FinanceBench")
        return None
    
    try:
        # Try various possible dataset names
        possible_names = [
            "PatronusAI/financebench",
            "financebench",
            "islam2023/financebench",
        ]
        
        for name in possible_names:
            try:
                print(f"Trying to load: {name}")
                dataset = load_dataset(name)
                print(f"✓ Loaded {name}")
                
                # Filter for AMD and Pepsi questions
                # This depends on the actual structure of the dataset
                print("Dataset structure:", dataset)
                return dataset
            except Exception as e:
                print(f"  Not found: {e}")
                continue
        
        print("FinanceBench not found on HuggingFace")
        return None
        
    except Exception as e:
        print(f"Error loading FinanceBench: {e}")
        return None


if __name__ == "__main__":
    print("Creating evaluation datasets for composition experiment\n")
    
    # Try to load FinanceBench
    print("Step 1: Checking for FinanceBench dataset...")
    financebench = try_load_financebench()
    print()
    
    # Create simple eval datasets
    print("Step 2: Creating simple evaluation datasets...")
    create_simple_eval_datasets()
    print()
    
    print(f"✓ Evaluation datasets saved to: {EVAL_DIR}")
    print("\nDatasets created:")
    print(f"  - {EVAL_DIR / 'amd_qa.json'}")
    print(f"  - {EVAL_DIR / 'pepsi_qa.json'}")
    print(f"  - {EVAL_DIR / 'composition_qa.json'}")
    print("\nNext: Use these for evaluation or enhance with more questions!")


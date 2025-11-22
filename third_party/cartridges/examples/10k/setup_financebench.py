"""
Download and prepare AMD and PepsiCo 10-K documents from FinanceBench.

FinanceBench: https://github.com/patronus-ai/financebench
Contains 10-K, 10-Q, and 8-K filings with Q&A pairs for evaluation.
"""

import json
import os
import subprocess
from pathlib import Path
import PyPDF2

CARTRIDGES_DIR = Path(__file__).parent.parent.parent
DATA_DIR = CARTRIDGES_DIR / "data" / "10k"
FINANCEBENCH_DIR = CARTRIDGES_DIR / "data" / "financebench"

def clone_financebench():
    """Clone the FinanceBench repository."""
    if FINANCEBENCH_DIR.exists():
        print(f"FinanceBench already exists at {FINANCEBENCH_DIR}")
        return
    
    print("Cloning FinanceBench repository...")
    subprocess.run([
        "git", "clone",
        "https://github.com/patronus-ai/financebench.git",
        str(FINANCEBENCH_DIR)
    ], check=True)
    print(f"✓ Cloned to {FINANCEBENCH_DIR}")


def extract_pdf_text(pdf_path):
    """Extract text from PDF file."""
    text = []
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text.append(page.extract_text())
    return '\n'.join(text)


def find_company_documents(company_name):
    """Find 10-K documents for a specific company."""
    doc_info_file = FINANCEBENCH_DIR / "data" / "financebench_document_information.jsonl"
    
    if not doc_info_file.exists():
        raise FileNotFoundError(f"Document info file not found: {doc_info_file}")
    
    documents = []
    with open(doc_info_file, 'r') as f:
        for line in f:
            doc = json.loads(line)
            if company_name.lower() in doc.get('company', '').lower():
                if doc.get('doc_type') == '10-K':
                    documents.append(doc)
    
    return documents


def setup_10k_documents():
    """
    Extract AMD and PepsiCo 10-K documents directly from the FinanceBench `pdfs/` dir.

    We **do not** depend on document_information.jsonl anymore for selecting the PDFs.
    Instead, we scan `pdfs/` and pick a reasonable AMD / PepsiCo 10-K file by filename.
    """
    pdf_dir = FINANCEBENCH_DIR / "pdfs"
    if not pdf_dir.exists():
        print(f"⚠ PDF directory not found: {pdf_dir}")
        return

    def pick_pdf(company_substrings, require_10k: bool = True):
        candidates = []
        for pdf_path in pdf_dir.glob("*.pdf"):
            name = pdf_path.name.lower()
            if any(s in name for s in company_substrings):
                if not require_10k or ("10-k" in name or "10k" in name):
                    candidates.append(pdf_path)
        # Simple heuristic: pick the lexicographically last candidate (often latest year)
        candidates = sorted(candidates)
        return candidates[-1] if candidates else None

    print("\nSearching for AMD 10-K PDFs in pdfs/ ...")
    amd_pdf = pick_pdf(["amd", "advanced micro devices"], require_10k=False)
    if amd_pdf is None:
        print("⚠ No AMD PDF found by filename; please check pdfs/ manually.")
    else:
        print(f"  Using AMD PDF: {amd_pdf.name}")
        amd_text = extract_pdf_text(amd_pdf)
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        output_path = DATA_DIR / "amd_10k_financebench.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(amd_text)
        print(f"✓ Saved AMD text to {output_path} ({len(amd_text):,} characters)")

    print("\nSearching for PepsiCo 10-K PDFs in pdfs/ ...")
    pepsi_pdf = pick_pdf(["pepsico", "pepsi"], require_10k=False)
    if pepsi_pdf is None:
        print("⚠ No PepsiCo PDF found by filename; please check pdfs/ manually.")
    else:
        print(f"  Using PepsiCo PDF: {pepsi_pdf.name}")
        pepsi_text = extract_pdf_text(pepsi_pdf)
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        output_path = DATA_DIR / "pepsi_10k_financebench.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(pepsi_text)
        print(f"✓ Saved PepsiCo text to {output_path} ({len(pepsi_text):,} characters)")


def load_financebench_qa():
    """Load Q&A pairs from FinanceBench for AMD and PepsiCo."""
    qa_file = FINANCEBENCH_DIR / "data" / "financebench_open_source.jsonl"
    
    if not qa_file.exists():
        print(f"⚠ Q&A file not found: {qa_file}")
        return [], []
    
    amd_qa = []
    pepsi_qa = []
    
    with open(qa_file, 'r') as f:
        for line in f:
            qa = json.loads(line)
            company = qa.get('company', '').lower()
            
            if 'amd' in company:
                amd_qa.append(qa)
            elif 'pepsi' in company:
                pepsi_qa.append(qa)
    
    print(f"\nFound {len(amd_qa)} AMD Q&A pairs")
    print(f"Found {len(pepsi_qa)} PepsiCo Q&A pairs")
    
    # Save for evaluation
    if amd_qa:
        with open(DATA_DIR / "eval" / "amd_qa_financebench.json", 'w') as f:
            json.dump(amd_qa, f, indent=2)
        print(f"✓ Saved AMD Q&A to {DATA_DIR / 'eval' / 'amd_qa_financebench.json'}")
    
    if pepsi_qa:
        with open(DATA_DIR / "eval" / "pepsi_qa_financebench.json", 'w') as f:
            json.dump(pepsi_qa, f, indent=2)
        print(f"✓ Saved PepsiCo Q&A to {DATA_DIR / 'eval' / 'pepsi_qa_financebench.json'}")
    
    return amd_qa, pepsi_qa


if __name__ == "__main__":
    print("="*80)
    print("FINANCEBENCH SETUP FOR CARTRIDGE EXPERIMENT")
    print("="*80)
    
    # Create directories
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "eval").mkdir(exist_ok=True)
    
    # Step 1: Clone FinanceBench
    try:
        clone_financebench()
    except Exception as e:
        print(f"Error cloning FinanceBench: {e}")
        print("Please clone manually:")
        print(f"  git clone https://github.com/patronus-ai/financebench.git {FINANCEBENCH_DIR}")
        exit(1)
    
    # Step 2: Extract 10-K documents
    try:
        setup_10k_documents()
    except Exception as e:
        print(f"Error extracting documents: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 3: Load Q&A pairs
    try:
        load_financebench_qa()
    except Exception as e:
        print(f"Error loading Q&A pairs: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("✓ SETUP COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Update training configs to use:")
    print(f"     - {DATA_DIR / 'amd_10k_financebench.txt'}")
    print(f"     - {DATA_DIR / 'pepsi_10k_financebench.txt'}")
    print("2. Run synthesis and training pipeline")
    print("3. Evaluate using FinanceBench Q&A pairs")



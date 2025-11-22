import json
from pathlib import Path

# Load scored QA results from the repo-relative path
results_path = Path("data/10k/eval/tokasaurus_qa_results_scored.json")
with results_path.open("r", encoding="utf-8") as f:
    qa_list = json.load(f)

amd_only_correct = 0
amd_total = 0
pepsi_total = 0
pepsi_only_correct = 0
composed_amd_correct = 0
composed_pepsi_correct = 0

for q in qa_list:
    if q['config'] == 'amd_only':
        amd_only_correct += q['correct']
        amd_total += 1
    elif q['config'] == 'pepsi_only':
        pepsi_only_correct += q['correct']
        pepsi_total += 1
    elif q['config'] == 'both':
        if q['split'] == 'amd':
            composed_amd_correct += q['correct']
        else:
            composed_pepsi_correct += q['correct']
    


text = f'''
BASE RESULTS:
amd_only: {amd_only_correct}/{amd_total}
pepsi_only: {pepsi_only_correct}/{pepsi_total}

COMPOSED RESULTS:
amd: {composed_amd_correct}/{amd_total}
pepsi: {composed_pepsi_correct}/{pepsi_total}
'''

print(text)

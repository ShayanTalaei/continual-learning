---
noteId: "3db2e7a0a5f911f097f9d90551ec87df"
tags: []

---

# Fine-Tuning

To fine-tune using LoRA with custom dataset:
1. Install required Python library in `../../environment.yml` using conda. 
1. Make sure the dataset jsonl file exist in the `data` directory, with similar format to existing datasets.
1. Edit `script_train.sh` with dataset path and change training hyper-parameters if needed.
1. run `script_train.sh`.

```shell
bash script_train.sh
``` 
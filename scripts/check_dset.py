from tqdm import tqdm
import torch
import os
import sys
sys.path.append("/afs/cs.stanford.edu/u/bcabrown/continual-learning")
sys.path.append("/afs/cs.stanford.edu/u/bcabrown/continual-learning/third_party/cartridges")
os.environ["CARTRIDGES_DIR"] = "/scratch/m000122/bcabrown/continual-learning/third_party/cartridges"
os.environ["CARTRIDGES_OUTPUT_DIR"] = "/scratch/m000122/bcabrown/continual-learning/third_party/cartridges/output"
from third_party.cartridges.cartridges.datasets import read_conversations

path = "/matx/u/bcabrown/shayan_memory/data/batman_reflection_finer_filtered.jsonl"

convos = read_conversations(path)

breakpoint()
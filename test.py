import torch
import transformers
import datasets

print("=" * 50)
print("Environment Test")
print("=" * 50)

print("PyTorch Version:", torch.__version__)
print("Transformers Version:", transformers.__version__)
print("Datasets Version:", datasets.__version__)

print("\nEverything is working correctly!")
from turngpt import TurnGPT
import torch
# Simple Plot
import matplotlib.pyplot as plt

# Fresh Initialization
# Default values are used, so gpt2.0 is the pretrained model
model = TurnGPT()

model.init_tokenizer()
model.initialize_special_embeddings()

# Example use
turn_list = [
    "Hello there I basically had the worst day of my life",
    "Oh no, what happened?",
    "Do you want the long or the short story?",
]
# Get trp from a text list
out = model.string_list_to_trp(turn_list)

print(f"Probabilities:\n    {out['trp_probs'][0]}\nTokens:\n    {out['tokens'][0]}")    

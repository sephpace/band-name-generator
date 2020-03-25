
import torch
import torch.nn as nn
import yaml

from data import load_alphabet
from models import BandNameGenerator


# --- Process the data ---

# Load the data
with open('data/band_names.yml', 'r') as file:
    data = yaml.full_load(file.read())

target_text = []
for key in data:
    target_text.extend(data[key])
target_text = sorted(list(set(target_text)))

# Load the alphabet
alphabet = load_alphabet()

# --- Set up the Model ---
bng = BandNameGenerator(len(alphabet))

# --- Train the Model ---

import pickle
from utils.encoder import MultiHotEncoder

with open('lapprediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('f1_data.pkl', 'rb') as f:
    comp = pickle.load(f)

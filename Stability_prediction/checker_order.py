import pickle
import pandas as pd

df = pd.read_csv("pandora.csv")

with open("mut_types_dict.pkl", "rb") as f:
    data = pickle.load(f)
    for key in data.keys():
        print(key)
        assert len(df[df['new_name'].str.rsplit('_', n=1).str[0] == key]) == len(data[key]), f"Mismatch for {key}"
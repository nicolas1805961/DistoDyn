import csv
import pandas as pd
import pickle

def parse_rec_file(input_path, output_path, df_concat, mega_splits):
    with open(input_path, "r") as f:
        content = f.read()

    # Split on separators
    entries = content.strip().split("--")
    seen_sequences = {}
    wt_sequences = -1

    parsed_rows = []

    name_to_split_wt = dict(zip(df_concat["WT_name"], df_concat["split"]))
    name_to_split = dict(zip(df_concat["name"], df_concat["split"]))

    for i, entry in enumerate(entries):
        entry = entry.strip()
        if not entry:
            continue

        data = {}
        for line in entry.split("\n"):
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
        
        if data['aa_seq_wt'] not in seen_sequences:
            wt_sequences += 1
            seen_sequences[data['aa_seq_wt']] = (wt_sequences, 0)

        name = data.get("WT_name", "")
        split = name_to_split_wt.get(name, "")
        if split == "":
            name = data.get("WT_name", "")
            split = name_to_split.get(name, "")
            if split == "":
                if name in mega_splits['train']:
                    split = 'train'
                elif name in mega_splits['val']:
                    split = 'validation'
                elif name in mega_splits['test']:
                    split = 'test'

        # Build a row with required columns
        row = {
            "name": data.get("name", ""),
            "dG_ML": data.get("dGmean", ""),
            "ddG_ML": data.get("ddGmean", ""),
            "mut_type": data.get("mut_type", ""),
            "WT_name": data.get("WT_name", ""),
            "aa_seq": data.get("aa_seq", ""),
            "wt_seq": data.get("aa_seq_wt", ""),
            "split": split,
            #"split": data.get("split", ""),
            "new_name": f"protein_{seen_sequences[data['aa_seq_wt']][0]}_{seen_sequences[data['aa_seq_wt']][1]}",
        }
        parsed_rows.append(row)
        seen_sequences[data['aa_seq_wt']] = (seen_sequences[data['aa_seq_wt']][0], seen_sequences[data['aa_seq_wt']][1] + 1)

        

    # Save CSV
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "name", "dG_ML", "ddG_ML", "mut_type", "WT_name", "aa_seq", "wt_seq", "split", "new_name"
        ])
        writer.writeheader()
        writer.writerows(parsed_rows)

    print(f"Saved CSV to {output_path}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse .rec file to CSV")
    parser.add_argument("input", help="Path to .rec file")
    parser.add_argument("output", help="Output CSV file")

    df_train = pd.read_csv("mega_train.csv")
    df_train["split"] = 'train'
    df_val = pd.read_csv("mega_val.csv")
    df_val["split"] = 'validation'
    df_test = pd.read_csv("mega_test.csv")
    df_test["split"] = 'test'
    df_concat = pd.concat([df_train, df_val, df_test], join="inner", ignore_index=True)

    with open("mega_splits.pkl", "rb") as f:
        mega_splits = pickle.load(f)

    args = parser.parse_args()
    parse_rec_file(args.input, args.output, df_concat, mega_splits)

"""
RNA-FM 特征提取完整解决方案
"""
import torch
import fm
import numpy as np
from Bio import SeqIO
import random
import os
from tqdm import tqdm



def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()


model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model.eval()


DATA_DIR = "datasets/"
SAVE_DIR = "datasets/features"
os.makedirs(SAVE_DIR, exist_ok=True)

file_groups = {
    "train_pos": ["Train_mRNA_mode_pos.fasta"],
    "train_neg": ["Train_mRNA_mode_neg.fasta"],
    "test_pos": ["Test_mRNA_mode_pos.fasta"],
    "test_neg": ["Test_mRNA_mode_neg.fasta"]
}


def process_sequence(seq):
    seq = seq.upper().replace('T', 'U')
    valid_chars = {'A', 'U', 'C', 'G', 'N'}
    seq = ''.join([c if c in valid_chars else 'N' for c in seq])
    return seq


def read_fasta(file_path):
    records = []
    for record in SeqIO.parse(file_path, "fasta"):
        processed_seq = process_sequence(str(record.seq))
        records.append((record.id, processed_seq))
    return records


def extract_features(sequences, batch_size=32):
    all_features = []

    for i in tqdm(range(0, len(sequences), batch_size),
                  desc="Processing batches",
                  unit="batch"):
        batch = sequences[i:i + batch_size]

        try:
          
            _, _, tokens = batch_converter(batch)
           
            with torch.no_grad():
                outputs = model(tokens, repr_layers=[12])
           
            embeddings = outputs["representations"][12]  
            pooled = torch.mean(embeddings, dim=1) 

            all_features.append(pooled.cpu().numpy())
        except Exception as e:
            print(f"Error processing batch {i // batch_size}: {str(e)}")

    return np.concatenate(all_features, axis=0)


def process_dataset():

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "rna_fm_weights_RNA.pth"))

    for group, files in file_groups.items():
        print(f"\nProcessing {group}...")

        all_sequences = []
        for fname in files:
            file_path = os.path.join(DATA_DIR, fname)
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found")
                continue

            print(f"Reading {fname}...")
            sequences = read_fasta(file_path)
            print(f"Loaded {len(sequences)} sequences")
            all_sequences.extend(sequences)

        if len(all_sequences) == 0:
            print("No sequences found, skipping...")
            continue

        features = extract_features(all_sequences)

        save_path = os.path.join(SAVE_DIR, f"{group}_features.npy")
        np.save(save_path, features)
        print(f"Saved {features.shape} features to {save_path}")


if __name__ == "__main__":
    process_dataset()
    print("\nFeature extraction completed!")
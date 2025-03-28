import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, default='amazon-cds')
args = parser.parse_args()
dataset = args.dataset



import pandas as pd

train_df = pd.read_csv(f'./data/{dataset}/train_fpsr.tsv', sep='\t', names=['user_id', 'item_id'])
valid_df = pd.read_csv(f'./data/{dataset}/valid_fpsr.tsv', sep='\t', names=['user_id', 'item_id'])
test_df = pd.read_csv(f'./data/{dataset}/test_fpsr.tsv', sep='\t', names=['user_id', 'item_id'])

full_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
full_df = full_df.drop_duplicates(subset=['user_id', 'item_id'])
full_df.to_csv(f'./data/{dataset}/{dataset}.tsv', sep='\t', index=False, header=False)

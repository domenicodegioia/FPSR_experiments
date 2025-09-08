import pandas as pd
import argparse


def process_dataset(dataset_name, train_file, test_file):
    train_df = pd.read_csv(train_file, sep='\t', header=None, names=['user', 'item', 'rating'])
    test_df = pd.read_csv(test_file, sep='\t', header=None, names=['user', 'item', 'rating'])

    item_popularity = train_df['item'].value_counts().reset_index()
    item_popularity.columns = ['item', 'popularity']
    item_popularity = item_popularity.sort_values(by='popularity', ascending=False)

    num_items = len(item_popularity)
    head_size = int(num_items * 0.10)

    head_items = item_popularity.head(head_size)['item'].tolist()
    tail_items = item_popularity.tail(num_items - head_size)['item'].tolist()

    print(f"Total items: {num_items}")
    print(f"Head items: {len(head_items)} ({len(head_items)/num_items * 100:.2f}%)")
    print(f"Tail items: {len(tail_items)} ({len(tail_items)/num_items * 100:.2f}%)")

    test_head_df = test_df[test_df['item'].isin(head_items)]
    test_tail_df = test_df[test_df['item'].isin(tail_items)]

    test_head_df.to_csv(f'data/{dataset_name}/splitting/0/test_head.tsv', sep='\t', header=False, index=False)
    print(f"File 'data/{dataset_name}/splitting/0/test_head.tsv' created.")
    test_tail_df.to_csv(f'data/{dataset_name}/splitting/0/test_tail.tsv', sep='\t', header=False, index=False)
    print(f"File 'data/{dataset_name}/splitting/0/test_tail.tsv' created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name (es. ml-1m)")
    args = parser.parse_args()

    dataset_name = args.dataset

    train_file_path = f'data/{dataset_name}/splitting/0/0/train.tsv'
    test_file_path = f'data/{dataset_name}/splitting/0/test.tsv'

    process_dataset(dataset_name, train_file_path, test_file_path)
import os

datasets = ['yelp2018', 'douban', 'gowalla', 'amazon-cds']
models = ['fpsr', 'itemknn']

base_dir = 'heatmap'

if not os.path.exists(base_dir):
    os.mkdir(base_dir)

for dataset in datasets:
    dataset_dir = os.path.join(base_dir, dataset)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    for model in models:
        model_dir = os.path.join(dataset_dir, model)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
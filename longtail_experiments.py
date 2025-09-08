from elliot.run import run_experiment

configs = [
    # "longtail_overall_amazon-cds",
    # "longtail_head_amazon-cds",
    # "longtail_tail_amazon-cds",
    # "longtail_overall_douban",
    # "longtail_head_douban",
    # "longtail_tail_douban",
    # "longtail_overall_gowalla",
    # "longtail_head_gowalla",
    # "longtail_tail_gowalla",
    # "longtail_overall_yelp2018",
    "longtail_head_yelp2018",
    "longtail_tail_yelp2018",
]

for config in configs:
    run_experiment(f"config_files/{config}.yml")

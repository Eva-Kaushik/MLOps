#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse

from pathlib import Path
import os
import numpy as np
import pandas as pd

# In[2]:


TARGET_COL = "cost"

NUMERIC_COLS = [
    "distance",
    "dropoff_latitude",
    "dropoff_longitude",
    "passengers",
    "pickup_latitude",
    "pickup_longitude",
    "pickup_weekday",
    "pickup_month",
    "pickup_monthday",
    "pickup_hour",
    "pickup_minute",
    "pickup_second",
    "dropoff_weekday",
    "dropoff_month",
    "dropoff_monthday",
    "dropoff_hour",
    "dropoff_minute",
    "dropoff_second",
]

CAT_NOM_COLS = [
    "store_forward",
    "vendor",
]

CAT_ORD_COLS = [
]

# In[3]:


class MyArgs:
    def __init__(self, /, **kwargs):
        self.__dict__.update(kwargs)

args = MyArgs(
            raw_data = "../../data/", 
            train_data = "/tmp/prep/train",
            val_data = "/tmp/prep/val",
            test_data = "/tmp/prep/test",
            )

os.makedirs(args.train_data, exist_ok = True)
os.makedirs(args.val_data, exist_ok = True)
os.makedirs(args.test_data, exist_ok = True)


# In[6]:


import argparse
import os
import sys
import wandb

os.environ["WANDB_MODE"] = "dryrun"

def main(args):
    pass

if __name__ == "__main__":
    if wandb.run is None:
        wandb.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", type=str, help="Raw data path")
    parser.add_argument("--train_data", type=str, help="Train dataset output path")
    parser.add_argument("--val_data", type=str, help="Val dataset output path")
    parser.add_argument("--test_data", type=str, help="Test dataset path")
    args, unknown = parser.parse_known_args()

    wandb.config.update(args)

    lines = [
        f"Raw data path: {args.raw_data}",
        f"Train dataset output path: {args.train_data}",
        f"Val dataset output path: {args.val_data}",
        f"Test dataset path: {args.test_data}",
    ]
    for line in lines:
        print(line)

    main(args)
    wandb.finish()


# In[7]:


ls "/tmp/prep/train" 

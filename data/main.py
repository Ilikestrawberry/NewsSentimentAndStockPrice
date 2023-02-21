import os
import sys

import pandas as pd
import numpy as np
import multiprocessing as mp
from multiprocessing import freeze_support

from pandas_parallel_apply import DataFrameParallel

from crawl import Crawl
from context import context
import argparse

if __name__ == "__main__":
    freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("--company_name", default="하이닉스")
    parser.add_argument("--sort", default="date")
    args = parser.parse_args()

    print(" ### Api Call Start ###")
    cw = Crawl(client_id="", client_secret="", args=args)
    result = cw(query=args.company_name, number=3000)
    result.to_pickle(f"./newsdata/{args.company_name}_{args.sort}.pkl")

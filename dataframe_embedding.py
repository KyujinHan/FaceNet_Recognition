# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 20:28:29 2021

@author: Made by  Kyujin Han
"""

import numpy as np
import pandas as pd
import os

df = pd.read_csv("./df_all_encoder_10.csv")
embedding = np.load("./embedding_all_128_L2.npy")

if len(df)==len(embedding) and len(df)==len(os.listdir("./lfw_over_10")):
    print(len(df))
    print("길이가 같다")
else:
    print(len(df))
    print(len(embedding))
    print(len(os.listdir("./lfw_over_10")))
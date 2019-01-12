# -*- coding: utf-8 -*-
import pyprind
import pandas as pd
import os
pbar= pyprind.progBar(50000)
labels = {'pos':1,'neg':0}
df = pd.DataFrame()

for s in ('test', 'train'):
    for l in ('pos','neg'):
        path = './aclImdb/%s/%s' % (s,l)
        for file in os.listdir(path):
            with open(os.path.join(path,file),'r') as infile:
                txt = infile.read()
            df = df.append([[txt,labels[1]]],ignore_index = True)
            pbar.update()
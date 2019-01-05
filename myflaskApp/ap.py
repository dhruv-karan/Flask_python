# -*- coding: utf-8 -*-
import pickle
import os
dest = os.path.join('GenderClassifier','pkl_objects')

if not os.path.exists(dest):
    os.makedirs(dest)

pickle.dump(clf,open(os.path.join(dest,'classifier.pkl '),'wb'),protocol=4)



cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(os.path.join(cur_dir,'pkl_objects'), 'rb'))

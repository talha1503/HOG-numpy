from hog_features import calculate_hog_features
import csv
import os
import numpy as np
import pandas as pd

image_directory = '../images'

df = pd.DataFrame(pd.np.empty((0, 3781)))
cols = ['image_id']
feature_cols = list(range(3780))
cols.extend(feature_cols)

df.columns = cols


with open('dataset.csv','a',newline='') as file:
    wr = csv.writer(file,dialect='excel')
    for index,image_path in enumerate(os.listdir(image_directory)):
        row = [index]
        feature_vector = np.ravel(calculate_hog_features(image_directory+'/'+image_path))
        row.extend(feature_vector)
        wr.writerow(row)



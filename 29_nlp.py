import numpy as np;
import matplotlib.pyplot as plt
import pandas as pd
import re

#quoting = 3 ignore double quotes
dataset = pd.read_csv('datafiles/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

#cleaning dataset
review = re.sub('[^a-zA-Z]',' ',dataset['Review'][0])
review = review.lower()


print(review)
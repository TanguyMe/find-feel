

# Import Pandas
import pandas as pd

# Import Split Tool
from sklearn.model_selection import train_test_split

# Set Relative Path
import sys
sys.path.append("../src")

# Import data
df_dataK = pd.read_csv(r'./data/d03_cleaned_data/CleanKaggle.csv')

# Split Train Test
train, test = train_test_split(df_dataK, test_size=0.2)

# Export to CSV
test.to_csv("./data/d03_cleaned_data/datall_test.csv",index=False)

# Remove cache
del test

# Import data
df_dataW = pd.read_csv(r'./data/d03_cleaned_data/CleanDataworld.csv')

# Merge dataframe
frames = [train , df_dataW]
df_datall = pd.concat(frames)

# Remove cache
del df_dataW , df_dataK

# Export to CSV
df_datall.to_csv("./data/d03_cleaned_data/datall.csv",index=False)

# Remove cache
del df_datall
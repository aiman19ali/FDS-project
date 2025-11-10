import pandas as pd
df = pd.read_csv("C:/Users/hp/OneDrive/Desktop/FDS/superstore.csv", low_memory=False)
print("Shape:", df.shape)
print(df.columns.tolist())
print(df.head(10).T)  # transpose so you can easily read column values

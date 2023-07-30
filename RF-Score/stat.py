# %%
import pandas as pd
import numpy as np

# %%
df_test = pd.read_csv("./results_20runs/test.csv")
df_test_random = pd.read_csv("./results_20runs/test_random.csv")
df_test_scaffold = pd.read_csv("./results_20runs/test_scaffold.csv")
df_test_sequence = pd.read_csv("./results_20runs/test_sequence.csv")

# %%
print("test")
print(df_test.describe().round(3))

print("test_random")
print(df_test_random.describe().round(3))

print("test_scaffold")
print(df_test_scaffold.describe().round(3))

print("test_sequence")
print(df_test_sequence.describe().round(3))


# %%
df_test = pd.read_csv("./results/test.csv")
df_test_random = pd.read_csv("./results/test_random.csv")
df_test_scaffold = pd.read_csv("./results/test_scaffold.csv")
df_test_sequence = pd.read_csv("./results/test_sequence.csv")

# %%
print("test")
print(df_test.describe().round(3))

print("test_random")
print(df_test_random.describe().round(3))

print("test_scaffold")
print(df_test_scaffold.describe().round(3))

print("test_sequence")
print(df_test_sequence.describe().round(3))

# %%
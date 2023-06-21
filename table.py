import pandas as pd

# Load the data from the CSV file
data = pd.read_csv("results.csv")  # replace 'filename.csv' with the actual file name

# Calculate the relative mean gap and its standard deviation
data["Relative Mean Gap"] = (
    abs(data["Average reward"] - 18440.55617163667) / 18440.55617163667
)
data["Relative Standard Deviation"] = (
    data["Standard deviation"] / data["Average reward"]
)

# Print the results in LaTeX table format
with pd.option_context("max_colwidth", 1000):
    print(data.to_latex(index=False, float_format="%.6f"))

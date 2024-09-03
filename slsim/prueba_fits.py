"""#from astropy.table import Table

#cluster_catalog = Table.read("/absolute/path/to/clusters_example.fits")
#members_catalog = Table.read("/absolute/path/to/members_example.fits")

#cluster_catalog = Table.read("C:/Users/pinov/slsim_pruebas/data/redMaPPer/clusters_example.fits")
#members_catalog = Table.read("C:/Users/pinov/slsim_pruebas/data/redMaPPer/members_example.fits")

# Check the data structure
#print(cluster_catalog)
#print(members_catalog)

from astropy.io import fits

# Load the FITS file
#file_path = '/path/to/your/clusters_example.fits'  # Replace with your actual file path
file_path = 'C:/Users/pinov/slsim_pruebas/data/redMaPPer/clusters_example.fits'  # Replace with your actual file path
hdul = fits.open(file_path)

# Print the header information to see the structure
hdul.info()

# Assuming the data is in the first HDU after the primary HDU
data = hdul[1].data

# Get all column names
columns = data.columns.names
print(f"Number of columns: {len(columns)}")
print("Column names:", columns)

# View the first few rows of the table to understand the structure
print("\nFirst few rows:")
print(data[:5])  # Print the first 5 rows

# Check if 'zlambda' is a column in the data
if 'zlambda' in columns:
    # Get the values of 'zlambda' column
    zlambda_values = data['zlambda']
    print("\n'zlambda' column values:")
    print(zlambda_values)
else:
    print("\n'zlambda' column is not found in the FITS file.")

# Close the FITS file
hdul.close()
"""

from astropy.table import Table
import pandas as pd

# Load the FITS files
file_path = "C:/Users/pinov/slsim_pruebas/data/redMaPPer/clusters_example.fits"
cluster_catalog = Table.read(file_path)

file_path2 = "C:/Users/pinov/slsim_pruebas/data/redMaPPer/members_example.fits"
member_catalog = Table.read(file_path2)

# Convert the Astropy Table to a pandas DataFrame
df = cluster_catalog.to_pandas()
df1 = member_catalog.to_pandas()

# Search for a specific value '0.7447' in cluster_catalog
specific_value = 0.7447
matches = df.isin([specific_value]).any(axis=1)
matched_rows = df[matches]

# Get the number of rows in the cluster_catalog
num_rows = len(df)

# Output the results
print(f"Number of rows in the FITS file (cluster_catalog): {num_rows}")
print("Rows containing the specific value '0.7447':")
print(matched_rows)

# Save the cluster_catalog DataFrame to a CSV file
csv_file_path = "C:/Users/pinov/slsim_pruebas/data/redMaPPer/clusters_example.csv"
df.to_csv(csv_file_path, index=False)

# Save the member_catalog DataFrame to a CSV file
csv_file_path2 = "C:/Users/pinov/slsim_pruebas/data/redMaPPer/members_example.csv"
df1.to_csv(csv_file_path2, index=False)

import pandas as pd
import geopandas as gpd
import numpy as np
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from libpysal.weights import DistanceBand
from libpysal.weights import W
from esda.moran import Moran
import os
import utils

geographic_crs ='EPSG:4326'
data_dirpath = os.path.join(utils.ESRD_ROOT, 'Data')
county_centroids_url = 'https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt'
county_centroids_filepath = os.path.join(data_dirpath, 'uniform_pipeline_v3/county_centroids.csv')
if not os.path.exists(county_centroids_filepath):
    county_centroids = pd.read_csv(county_centroids_url)
    county_centroids['STATEFP'] = county_centroids['STATEFP'].astype(str).str.zfill(2)
    county_centroids['COUNTYFP'] = county_centroids['COUNTYFP'].astype(str).str.zfill(3)
    county_centroids['GEOID'] = county_centroids['STATEFP'] + county_centroids['COUNTYFP']
    county_centroids.to_csv(county_centroids_filepath, index=False)

county_centroids = pd.read_csv(county_centroids_filepath)
county_centroids['GEOID'] = county_centroids['GEOID'].astype(str).str.zfill(5)
county_centroids['LONGITUDE'] = county_centroids['LONGITUDE'].astype(float)
county_centroids['LATITUDE'] = county_centroids['LATITUDE'].astype(float)
county_centroids.rename(columns={'LONGITUDE': 'centroid_lon', 'LATITUDE': 'centroid_lat'}, inplace=True)
print(county_centroids[(county_centroids.centroid_lon.isna()) | (county_centroids.centroid_lat.isna())])

cutoff = 100.0
dialysis_burden_access_dirpath = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/dialysis_burden_access_2023')
dfbi_df = pd.read_csv(os.path.join(dialysis_burden_access_dirpath, f'dialysis_burden_all_facilities_range_{cutoff}miles.csv'))
dfbi_df['GEOID'] = dfbi_df['facility_county_fips'].astype(str).str.zfill(5)
dfbi_df = dfbi_df.merge(county_centroids[['GEOID', 'centroid_lon', 'centroid_lat']], on='GEOID')
dfbi_df.rename(columns={'dialysis_burden_Crude':'DFBI'}, inplace=True)
print(dfbi_df.head())


# -----------------------------
# Load your dataframe
# -----------------------------
df = dfbi_df.copy()

# Coordinates in radians for haversine
coords = df[['centroid_lat', 'centroid_lon']].values
coords_rad = np.radians(coords)

print(coords)
# -----------------------------
# Compute Haversine distance matrix
# -----------------------------
dist_matrix = haversine_distances(coords_rad)

# Convert radians → kilometers
earth_radius_miles = 3959
dist_miles = haversine_distances(coords_rad) * earth_radius_miles

# Distance-band threshold in miles
threshold_miles = 100.0

# Build neighbors/weights from distance matrix
n = len(df)
neighbors = {}
weights = {}

for i in range(n):
    js = np.where((dist_miles[i] <= threshold_miles) & (dist_miles[i] > 0))[0]
    neighbors[i] = js.tolist()
    weights[i] = [1.0] * len(js)

# Check for isolates (counties with 0 neighbors)
isolates = [i for i, nbrs in neighbors.items() if len(nbrs) == 0]
print(f"Isolates (0 neighbors) at {threshold_miles} miles: {len(isolates)}")

# If too many isolates, increase threshold (e.g., 150–300 miles)
# threshold_miles = 200.0  # rerun neighbors if needed

w = W(neighbors, weights)
w.transform = "R"  # row-standardize

# Moran’s I (global)
m_dfbi = Moran(df["DFBI"].to_numpy(), w, permutations=999)

print(f"DFBI Moran's I = {m_dfbi.I:.4f}")
print(f"p_sim = {m_dfbi.p_sim:.4g}  (based on {m_dfbi.permutations} permutations)")
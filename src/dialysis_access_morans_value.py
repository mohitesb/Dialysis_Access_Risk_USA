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
import matplotlib.pyplot as plt
from shapely import affinity


geographic_crs ='EPSG:4326'
plot_crs = 'EPSG:2163'
alaska_crs = 'EPSG:3338'
math_crs = 'EPSG:3857'
safegraph_dirpath = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/weekly_patterns')

counties_filepath = os.path.join(utils.ESRD_ROOT, 'Data/County_Shapefiles/tl_2020_us_county.shp')

counties = gpd.read_file(counties_filepath)
counties['GEOID'] = counties.GEOID.astype(str).str.zfill(5)

if counties.crs is None:
    counties.crs = geographic_crs
else:
    counties.to_crs(geographic_crs, inplace=True)
counties = counties.sort_values(by=['GEOID'])
n = len(counties.GEOID)
states = gpd.read_file(os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/state_boundaries/tl_2024_us_state.shp'))
states['STATEFP'] = states.STATEFP.astype(str).str.zfill(2)
counties['STATEFP'] = counties.STATEFP.astype(str).str.zfill(2)
states.rename(columns={'NAME': 'STNAME'}, inplace=True)
state_mapping = dict(zip(states['STATEFP'], states['STNAME']))
state_fips_mapping = dict(zip(states['STNAME'], states['STATEFP']))

counties = counties.merge(states[['STATEFP', 'STUSPS', 'STNAME']], on='STATEFP')


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
dpas_df = pd.read_csv(os.path.join(dialysis_burden_access_dirpath, f'dialysis_access_all_facilities_range_{cutoff}miles.csv'))
dpas_df['GEOID'] = dpas_df['county_fips'].astype(str).str.zfill(5)
dpas_df = dpas_df.merge(county_centroids[['GEOID', 'centroid_lon', 'centroid_lat']], on='GEOID')
dpas_df.rename(columns={'dialysis_access_Crude':'DPAS', 'dialysis_access_pct_rank':'DPAS_RANK'}, inplace=True)

print(dpas_df.describe())
lightgray_color = '#d3d3d3'
# -----------------------------
# Load your dataframe
# -----------------------------
df = dpas_df.copy()

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
print(f'Isolates (0 neighbors) at {threshold_miles} miles: {len(isolates)}')

# If too many isolates, increase threshold (e.g., 150–300 miles)
# threshold_miles = 200.0  # rerun neighbors if needed

w = W(neighbors, weights)
w.transform = 'R'  # row-standardize

# Moran’s I (global)
m_dpas = Moran(df['DPAS'].to_numpy(), w, permutations=999)

print(f'DPAS Moran\'s I = {m_dpas.I:.4f}')
print(f'p_sim = {m_dpas.p_sim:.4g}  (based on {m_dpas.permutations} permutations)')



from esda.moran import Moran_Local
lisa = Moran_Local(df['DPAS'].to_numpy(), w, permutations=999)

# Add results to dataframe
df['local_I'] = lisa.Is
df['p_value'] = lisa.p_sim
df['z_score'] = lisa.z_sim
df['quadrant'] = lisa.q
cluster_labels = {
    1: 'High-High',
    2: 'Low-High',
    3: 'Low-Low',
    4: 'High-Low'
}

df['cluster_type'] = df['quadrant'].map(cluster_labels)
significant = df['p_value'] < 0.05
df['significant_cluster'] = np.where(significant, df['cluster_type'], 'Not Significant')

hh_counties = df[(df['cluster_type'] == 'High-High') & (df['p_value'] < 0.05)]
print(f'High-High counties: {hh_counties[["GEOID", "DPAS", "DPAS_RANK", "local_I", "p_value"]]}')
ll_counties = df[(df['cluster_type'] == 'Low-Low') & (df['p_value'] < 0.05)]
print(f'Low-Low counties: {ll_counties[["GEOID", "DPAS", "DPAS_RANK", "local_I", "p_value"]]}')


high_dpas_clusters = df[
    (df['quadrant'] == 1) &
    (df['p_value'] < 0.05)
]

print(high_dpas_clusters[['GEOID','DPAS','DPAS_RANK','local_I','p_value']])

hh_mask = (df["quadrant"] == 1) & (df["p_value"] < 0.05)
hh_indices = np.where(hh_mask)[0]
hh_set = set(hh_indices)

hh_neighbors = {}
for i in hh_indices:
    nbrs = [j for j in w.neighbors[i] if j in hh_set]
    hh_neighbors[i] = nbrs
visited = set()
clusters = []

for node in hh_neighbors:
    if node not in visited:
        stack = [node]
        component = []

        while stack:
            n = stack.pop()
            if n not in visited:
                visited.add(n)
                component.append(n)
                stack.extend(hh_neighbors[n])

        clusters.append(component)
        
df = df.merge(counties[['GEOID', 'NAME', 'STUSPS', 'STNAME']], on='GEOID')

df["cluster_id"] = -1

for i, cluster in enumerate(clusters):
    df.loc[df.index[cluster], "cluster_id"] = i + 1

cluster_colors = ['#e31a1c', '#ae017e', '#cc4c02', '#ce1256', ]
cluster_color_map = dict(zip([i for i in range(1, len(cluster_colors)+1)], cluster_colors))
df['cluster_color'] = df['cluster_id'].map(lambda x: cluster_color_map[x] if x in cluster_color_map and pd.notnull(x) else lightgray_color)

df.sort_values(by=['cluster_id'], inplace=True)
df.to_csv(os.path.join(dialysis_burden_access_dirpath, 'dialysis_access_high_low_clusters.csv'), index=False)
print(df[['NAME', 'STNAME','DPAS','DPAS_RANK','p_value','cluster_id']])
print(f'unique clusters: {df["cluster_id"].unique()}')

for cluster_id_val in df["cluster_id"].unique():
    curr_df = df.loc[df["cluster_id"] == cluster_id_val]
    print(f'cluster_id: {cluster_id_val}, states: {curr_df.STNAME.unique()}, # counties: {curr_df[["NAME", "STNAME"]].nunique()}')

geographic_crs = 'EPSG:4326'
svi_df_filepath = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/SVI_2020_US_county/SVI2020_US_county.gdb')
svi_df = gpd.read_file(svi_df_filepath).to_crs(geographic_crs)
svi_df['FIPS'] = svi_df['FIPS'].astype(str).str.zfill(5)
svi_df['state_fips'] = svi_df['FIPS'].str[:2]

df = df.merge(svi_df[['FIPS', 'geometry']], left_on='GEOID', right_on='FIPS')
df = gpd.GeoDataFrame(df, geometry='geometry', crs=geographic_crs)
df['state_fips'] = df['FIPS'].str[:2].astype(int)
svi_df['state_fips'] = svi_df['state_fips'].astype(int)

colors = ['#ca0020', '#92c5de', '#0571b0', '#f4a582']
colormap = dict(zip(
    [1, 2, 3, 4], colors
))
df['colors'] = df['quadrant'].map(lambda x: colormap[x] if x in colormap and pd.notnull(x) else lightgray_color)
vulnerable_states = [
    'California', 'Oregon', 'Washington',
    'Vermont', 'New Hampshire', 'Maine', 'Rhode Island', 'Massachusetts', 'Connecticut',
    'New Jersey', 'New York', 'Pennsylvania',
    'Delaware', 'Maryland', 'Virginia', 'North Carolina', 'West Virginia',
    'Michigan'
]
vulnerable_states_fips = [int(state_fips_mapping[state]) for state in vulnerable_states]


plot_dirpath = os.path.join(dialysis_burden_access_dirpath, 'plots')
for curr_state_name in vulnerable_states:
    curr_state_fips = int(state_fips_mapping[curr_state_name])
    curr_svi_df = svi_df.loc[svi_df.state_fips == curr_state_fips]
    curr_df = df.loc[(df.state_fips == curr_state_fips) & (df.quadrant == 1) & (df.p_value < 0.05)]
    fig, ax = plt.subplots(figsize=(1, 1), constrained_layout=True)
    if len(curr_svi_df) > 0:
        curr_svi_df.plot(ax=ax, color=lightgray_color, edgecolor='black', linewidth=0.25)
    if len(curr_df) > 0:
        curr_df.plot(ax=ax, color=curr_df['colors'], edgecolor='black', linewidth=0.25)
    ax.set_axis_off()
    ax.margins(0.01)
    plt.savefig(os.path.join(plot_dirpath, f'dialysis_access_high_low_clusters_plot_{curr_state_name}.png'), dpi=600, bbox_inches="tight", pad_inches=0.1)
    plt.close()

# svi_df = svi_df.loc[svi_df.state_fips.isin(vulnerable_states_fips)]
# df = df.loc[(df.state_fips.isin(vulnerable_states_fips)) & (df.quadrant == 1) & (df.p_value < 0.05)]
df = df.loc[df.p_value < 0.05]

alaska_counties = svi_df.loc[svi_df.state_fips == 2].to_crs(crs=alaska_crs)
hawaii_counties = svi_df.loc[svi_df.state_fips == 15].to_crs(crs=geographic_crs)
usa_counties = svi_df.loc[(svi_df.state_fips != 2) & (svi_df.state_fips != 15)].to_crs(crs=geographic_crs)
usa_counties = usa_counties.loc[usa_counties.state_fips < 60]

usa_counties = usa_counties.to_crs(plot_crs)
alaska_counties = alaska_counties.to_crs(plot_crs)
hawaii_counties = hawaii_counties.to_crs(plot_crs)
alaska_counties['geometry'] = alaska_counties.geometry.apply(lambda g: affinity.scale(g, xfact=0.35, yfact=0.35, origin=(0, 0)))
alaska_counties['geometry'] = alaska_counties.geometry.apply(lambda g: affinity.translate(g, xoff=-1.5e6, yoff=-1.0e6))
alaska_counties['geometry'] = alaska_counties.geometry.apply(lambda g: affinity.rotate(g, angle=45, origin=(0, 0)))
hawaii_counties['geometry'] = hawaii_counties.geometry.apply(lambda g: affinity.scale(g, xfact=0.5, yfact=0.5, origin=(0, 0)))
hawaii_counties['geometry'] = hawaii_counties.geometry.apply(lambda g: affinity.translate(g, xoff=1.8e6, yoff=-1.3e6))
hawaii_counties['geometry'] = hawaii_counties.geometry.apply(lambda g: affinity.rotate(g, angle=20, origin=(0, 0)))

alaska_df = df.loc[df.state_fips == 2].to_crs(crs=alaska_crs)
hawaii_df = df.loc[df.state_fips == 15].to_crs(crs=geographic_crs)
usa_df = df.loc[(df.state_fips != 2) & (df.state_fips != 15)].to_crs(crs=geographic_crs)
usa_df = usa_df.loc[usa_df.state_fips < 60]
alaska_df = alaska_df.to_crs(plot_crs)
hawaii_df = hawaii_df.to_crs(plot_crs)
usa_df = usa_df.to_crs(plot_crs)
alaska_df['geometry'] = alaska_df.geometry.apply(lambda g: affinity.scale(g, xfact=0.35, yfact=0.35, origin=(0, 0)))
alaska_df['geometry'] = alaska_df.geometry.apply(lambda g: affinity.translate(g, xoff=-1.5e6, yoff=-1.0e6))
alaska_df['geometry'] = alaska_df.geometry.apply(lambda g: affinity.rotate(g, angle=45, origin=(0, 0)))
hawaii_df['geometry'] = hawaii_df.geometry.apply(lambda g: affinity.scale(g, xfact=0.5, yfact=0.5, origin=(0, 0)))
hawaii_df['geometry'] = hawaii_df.geometry.apply(lambda g: affinity.translate(g, xoff=1.8e6, yoff=-1.3e6))
hawaii_df['geometry'] = hawaii_df.geometry.apply(lambda g: affinity.rotate(g, angle=20, origin=(0, 0)))

print(f'svi_df: ', len(usa_counties), len(alaska_counties), len(hawaii_counties), len(svi_df))
print(f'df: ', len(usa_df), len(alaska_df), len(hawaii_df))

fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
hawaii_counties.plot(ax=ax, color=lightgray_color, edgecolor='black', linewidth=0.25)
hawaii_df.plot(ax=ax, color=lightgray_color, edgecolor='black', linewidth=0.25)
alaska_counties.plot(ax=ax, color=lightgray_color, edgecolor='black', linewidth=0.25)
alaska_df.plot(ax=ax, color=lightgray_color, edgecolor='black', linewidth=0.25)
usa_counties.plot(ax=ax, color=lightgray_color, edgecolor='black', linewidth=0.25)
usa_df.plot(ax=ax, color=df['colors'], edgecolor='black', linewidth=0.25)
ax.set_axis_off()
ax.margins(0.01)
plot_dirpath = os.path.join(dialysis_burden_access_dirpath, 'plots')
plt.savefig(os.path.join(plot_dirpath, 'dialysis_access_high_low_clusters_plot.png'), dpi=600, bbox_inches="tight", pad_inches=0.1)
plt.close()


fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
hawaii_counties.plot(ax=ax, color=lightgray_color, edgecolor='black', linewidth=0.25)
hawaii_df.plot(ax=ax, color=hawaii_df['cluster_color'], edgecolor='black', linewidth=0.25)
alaska_counties.plot(ax=ax, color=lightgray_color, edgecolor='black', linewidth=0.25)
alaska_df.plot(ax=ax, color=alaska_df['cluster_color'], edgecolor='black', linewidth=0.25)
usa_counties.plot(ax=ax, color=lightgray_color, edgecolor='black', linewidth=0.25)
usa_df.plot(ax=ax, color=usa_df['cluster_color'], edgecolor='black', linewidth=0.25)
ax.set_axis_off()
ax.margins(0.01)
plot_dirpath = os.path.join(dialysis_burden_access_dirpath, 'plots')
plt.savefig(os.path.join(plot_dirpath, 'dialysis_access_high_low_unique_clusters_plot.png'), dpi=600, bbox_inches="tight", pad_inches=0.1)
plt.close()


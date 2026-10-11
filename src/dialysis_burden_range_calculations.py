import pandas as pd
import geopandas as gpd
import os
import utils
import numpy as np
from tqdm.auto import tqdm
from dataloader import SafeGraph
from utils import get_good_cms_and_walking_patterns, haversine_vec
from dataloader import NeighborhoodExtraction
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

geographic_crs = 'EPSG:4326'

external_ssd_path = os.path.join(utils.EXTERNAL_SSD_ROOT, 'Data/')
cms_dialysis_filepath = os.path.join(utils.ESRD_ROOT, 'Data/CMS_Dialysis_Locations/DFC_FACILITY_arcgis_2024.geojson')
safegraph_dirpath = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/weekly_patterns')
cb_population_filepath_template = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/census_block_populations/{}.csv')

# esrd_patients_dirpath = os.path.join(utils.ESRD_ROOT, 'Data/esrdCountyLevelData')
esrd_patients_dirpath = os.path.join(utils.ESRD_ROOT, 'Data/esrdCountyLevelData/ESRD_Hemo_2023')
esrd_patients = {}
for fname in os.listdir(esrd_patients_dirpath):
    if fname.startswith('Incident') and fname.endswith('.csv'):
    # if fname.startswith('Prevalent ESRD Hemo') and fname.endswith('.csv'):
        filename = os.path.join(esrd_patients_dirpath, fname)
        df = pd.read_csv(filename, low_memory=False)
        if not 'FIPS' in df.columns:
            continue
        df['FIPS'] = df.FIPS.astype(str).str.zfill(5)
        df['Prevalent Count'] = df['Prevalent Count'].astype(str).str.replace(',', '').str.replace('*', '0').astype(float)
        patients = dict(zip(df.FIPS, df['Prevalent Count']))
        esrd_patients.update(patients)

cb_census_dirpath = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/census_block_populations')
county_pops = {}
tract_pops = []
for filename in os.listdir(cb_census_dirpath):
    if not filename.endswith('.csv'):
        continue
    county_fips = filename[:5]
    cb_census_filepath = os.path.join(cb_census_dirpath, filename)
    temp_df = pd.read_csv(cb_census_filepath)
    temp_df['state'] = temp_df.state.astype(str).str.zfill(2)
    temp_df['county'] = temp_df.county.astype(str).str.zfill(3)
    temp_df['tract'] = temp_df.tract.astype(str).str.zfill(6)
    temp_df['block'] = temp_df.block.astype(str).str.zfill(4)
    temp_df['cb_fips'] = temp_df.state + temp_df.county + temp_df.tract + temp_df.block
    county_pops[county_fips] = temp_df.P1_001N.sum()

    temp_df['tract_fips'] = temp_df['cb_fips'].str[:11]
    curr_tract_pops = temp_df[['tract_fips', 'P1_001N']].groupby('tract_fips').sum().reset_index()
    curr_tract_pops.rename(columns={'P1_001N': 'population'}, inplace=True)
    tract_pops.append(curr_tract_pops)
tract_pops = pd.concat(tract_pops)

esrd_rate = {county_fips:esrd_pop/county_pops[county_fips] for county_fips, esrd_pop in esrd_patients.items() if county_fips in county_pops and county_pops[county_fips] > 0 and esrd_pop > 0}
tract_pops['esrd_rate'] = tract_pops.tract_fips.str[:5].map(esrd_rate).fillna(0)
tract_pops['esrd_patients'] = tract_pops.population * tract_pops.esrd_rate
tract_pops = tract_pops.loc[tract_pops.esrd_patients > 0]



dialysis_burden_access_dirpath = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/dialysis_burden_access_2023')
os.makedirs(dialysis_burden_access_dirpath, exist_ok=True)

import json
def extract_working_hours(times):
    t = json.loads(times)
    x = 0
    n = 0
    for i in t.values():
        if len(i) == 0:
            continue
        end_time = i[0][1].split(':')
        end_time = int(end_time[0])*60 + int(end_time[1])
        start_time = i[0][0].split(':')
        start_time = int(start_time[0])*60 + int(start_time[1])
        t = end_time - start_time
        n += 1
        x += t
    return x / (60*7)

cms_df = gpd.read_file(cms_dialysis_filepath)
if cms_df.crs is None:
    cms_df = cms_df.to_crs(geographic_crs)
else:
    cms_df.to_crs(geographic_crs, inplace=True)
cms_df['CMS Certification Number (CCN)'] = cms_df['CMS Certification Number (CCN)'].astype(int)
cms_df['# of Dialysis Stations'] = cms_df['# of Dialysis Stations'].astype(float)

# Load urban area boundaries
urban_df = gpd.read_file(os.path.join(utils.ESRD_ROOT, 'Data/Urban/tl_2020_us_uac20.shp'))
if urban_df.crs is None:
    urban_df = urban_df.to_crs(geographic_crs)
else:
    urban_df.to_crs(geographic_crs, inplace=True)

urban_cms_ids = gpd.sjoin(cms_df, urban_df, how='inner', predicate='within')['CMS Certification Number (CCN)'].unique()
cms_df['urban'] = cms_df['CMS Certification Number (CCN)'].isin(urban_cms_ids)

sg = SafeGraph(safegraph_dirpath=safegraph_dirpath)
safegraph = sg.load_safegraph()
if safegraph.crs is None:
    safegraph = safegraph.to_crs(geographic_crs)
else:
    safegraph.to_crs(geographic_crs, inplace=True)




good_cms_df, good_safegraph = get_good_cms_and_walking_patterns(cms_df=cms_df, 
                                                                    safegraph=safegraph[safegraph.naics_code == 621492], 
                                                                    cms_id='CMS Certification Number (CCN)', 
                                                                    sg_id='placekey',
                                                                    cutoff=.1)

working_hours = {}
for _, row in good_safegraph[good_safegraph.open_hours.notna()].iterrows():
    if not row['placekey'] in working_hours:
        working_hours[row['placekey']] = []
    hours = extract_working_hours(row['open_hours'])
    working_hours[row['placekey']].append(hours)
working_hours = {k:np.median(np.array(v))*7 for k, v in working_hours.items()}
avg_working_hours = np.median(np.array(list(working_hours.values())))
good_cms_df['working_hours'] = good_cms_df.placekey.map(lambda x: working_hours.get(x, avg_working_hours))
cms_placekey_map = dict(zip(good_cms_df['CMS Certification Number (CCN)'], good_cms_df.placekey))

svi_df_filepath = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/SVI_2020_US_county/SVI2020_US_county.gdb')
svi_df = gpd.read_file(svi_df_filepath)
if svi_df.crs is None:
    svi_df = svi_df.to_crs(geographic_crs)
else:
    svi_df.to_crs(geographic_crs, inplace=True)
svi_df['FIPS'] = svi_df['FIPS'].astype(str).str.zfill(5)



# Add the county fips of facility
good_cms_df = cms_df.copy()
good_cms_df['placekey'] = good_cms_df['CMS Certification Number (CCN)'].map(lambda x: cms_placekey_map.get(x, None))
good_cms_df['working_hours'] = good_cms_df.placekey.map(lambda x: working_hours.get(x, avg_working_hours))

good_cms_df = gpd.sjoin(good_cms_df, svi_df[['FIPS', 'geometry']], predicate='within')
good_cms_df.rename(columns={'FIPS': 'facility_county_fips'}, inplace=True)
good_cms_df['max_sessions'] = good_cms_df['working_hours'] * good_cms_df['# of Dialysis Stations'] / 4.0


county_working_hours = good_cms_df[[
    'facility_county_fips', 'max_sessions'
]].groupby('facility_county_fips').sum().reset_index()

county_urban_working_hours = good_cms_df[[
    'facility_county_fips', 'max_sessions', 'urban'
]].groupby(['facility_county_fips', 'urban']).sum().reset_index()

county_working_hours = county_working_hours.merge(county_urban_working_hours[county_urban_working_hours.urban], on='facility_county_fips', how='left', suffixes=('', '_urban'))
county_working_hours = county_working_hours.merge(county_urban_working_hours[~county_urban_working_hours.urban], on='facility_county_fips', how='left', suffixes=('', '_rural'))

print(county_working_hours.head())

county_working_hours = county_working_hours.loc[county_working_hours.max_sessions > 0]


od_median_dirpath = os.path.join(external_ssd_path, 'uniform_pipeline_v3/baseline_conditions_node_level')
od_median_df = []
svi_df['GEOID'] = svi_df.FIPS.astype(str).str.zfill(5)

tract_svi_filepath = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/SVI_2020_US/SVI2020_US_tract.gdb')
tract_svi_df = gpd.read_file(tract_svi_filepath)
if tract_svi_df.crs is None:
    tract_svi_df = tract_svi_df.to_crs(geographic_crs)
else:
    tract_svi_df.to_crs(geographic_crs, inplace=True)
tract_svi_df['GEOID'] = tract_svi_df['FIPS'].astype(str).str.zfill(11)


reachable_tracts = {}
cutoff = 100.0
for county_fips, county_geometry in tqdm(zip(svi_df.FIPS, svi_df.geometry), total=len(svi_df), desc='Processing counties'):
    od_median_filepath = os.path.join(od_median_dirpath, f'{county_fips}/od_median.csv')
    if not os.path.exists(od_median_filepath):
        continue
    try:
        curr_od_median_df = pd.read_csv(od_median_filepath)
    except:
        print(f'Error reading od_median_filepath: {od_median_filepath}')
        continue

    if len(curr_od_median_df) == 0:
        continue
    curr_od_median_df['cbg_geoid'] = curr_od_median_df['cbg_geoid'].astype(str).str.zfill(12)
    curr_od_median_df['cms_id'] = curr_od_median_df['cms_id'].astype(int)
    od_median_df.append(curr_od_median_df)

    ne = NeighborhoodExtraction(tract_svi_df, county_geometry, cutoff=cutoff)
    study_area_geoids = ne.extract_study_area()
    reachable_tracts[county_fips] = study_area_geoids

    # if len(od_median_df) > 40:
    #     break

od_median_df = pd.concat(od_median_df)
od_median_df = od_median_df.loc[od_median_df.median_visits != 0]
od_median_df = od_median_df.loc[od_median_df.median_visits.notna()]

counties_with_records = set(list(od_median_df.cbg_geoid.str[:5].unique()))

od_median_df['tract_fips'] = od_median_df.cbg_geoid.str[:11]
tract_median_df = od_median_df[['tract_fips', 'cms_id', 'median_visits']].groupby(['tract_fips', 'cms_id']).agg({'median_visits': 'sum'}).reset_index()
tract_median_df['ratio'] = tract_median_df['median_visits'] / tract_median_df.groupby('tract_fips')['median_visits'].transform('sum')
tract_median_df = tract_median_df.merge(tract_pops[['tract_fips', 'esrd_patients']], on='tract_fips', how='right')


tract_median_df['adjusted_median_visits'] = tract_median_df['esrd_patients'] * tract_median_df['ratio']
tract_median_df = tract_median_df.merge(good_cms_df[['CMS Certification Number (CCN)', 'facility_county_fips']], left_on='cms_id', right_on='CMS Certification Number (CCN)', how='left')


# Calculate the percentage of adjusted median visits for cross-county (out-of-county) flows, by state fips.
# First, add state fips columns to make aggregation easier.
tract_median_df['state_fips'] = tract_median_df['tract_fips'].str[:2]
tract_median_df['county_fips'] = tract_median_df['tract_fips'].str[:5]
# Only rows with non-missing adjusted_median_visits (i.e., valid flow records)
valid = tract_median_df[tract_median_df['adjusted_median_visits'].notna()]
# Identify cross-county flows
cross_county = valid[valid['county_fips'] != valid['facility_county_fips']]
# Identify within-county flows
within_county = valid[valid['county_fips'] == valid['facility_county_fips']]

# Group by state_fips and sum adjusted_median_visits for both types
cross_counts = cross_county.groupby('state_fips')['adjusted_median_visits'].sum()
within_counts = within_county.groupby('state_fips')['adjusted_median_visits'].sum()
total_counts = valid.groupby('state_fips')['adjusted_median_visits'].sum()
# Calculate percentages: % of out-of-county (cross-county) visits at state level
cross_county_percentage_by_state = (cross_counts / total_counts * 100).fillna(0)

cross_counts_county = cross_county.groupby('county_fips')['adjusted_median_visits'].sum()
within_counts_county = within_county.groupby('county_fips')['adjusted_median_visits'].sum()
total_counts_county = valid.groupby('county_fips')['adjusted_median_visits'].sum()
cross_county_percentage_by_county = (cross_counts_county / total_counts_county * 100).fillna(0)


cross_county_percentage_by_state.sort_values(ascending=False, inplace=True)
print(f'cross_county_percentage_by_state: \n{cross_county_percentage_by_state}')
cross_county_percentage_by_state = cross_county_percentage_by_state.to_dict()

cross_county_percentage_by_county.sort_values(ascending=False, inplace=True)
print(f'cross_county_percentage_by_county: \n{cross_county_percentage_by_county}')
cross_county_percentage_by_county = cross_county_percentage_by_county.to_dict()

incoming_visits = tract_median_df[[
    'facility_county_fips', 'adjusted_median_visits'
]].groupby('facility_county_fips').sum().reset_index()



upper_bound_incoming_visits = {}
lower_bound_incoming_visits = {}
for _, row in tract_median_df[tract_median_df.median_visits.notna()].iterrows():
    facility_county_fips = row['facility_county_fips']
    county_fips = row['tract_fips'][:5]
    if not facility_county_fips in upper_bound_incoming_visits:
        upper_bound_incoming_visits[facility_county_fips] = 0
    if not facility_county_fips in lower_bound_incoming_visits:
        lower_bound_incoming_visits[facility_county_fips] = 0
    
    upper_bound_incoming_visits[facility_county_fips] += row['adjusted_median_visits']
    lower_bound_incoming_visits[facility_county_fips] += row['adjusted_median_visits']






county_centroids = pd.read_csv('https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt')
county_centroids['STATEFP'] = county_centroids['STATEFP'].astype(str).str.zfill(2)
county_centroids['COUNTYFP'] = county_centroids['COUNTYFP'].astype(str).str.zfill(3)
county_centroids['county_fips'] = county_centroids['STATEFP'] + county_centroids['COUNTYFP']
county_centroids['LATITUDE'] = county_centroids['LATITUDE'].astype(float)
county_centroids['LONGITUDE'] = county_centroids['LONGITUDE'].astype(float)
county_centroids_dict = {}
for _, row in county_centroids.iterrows():
    county_fips = row['county_fips']
    latitude = row['LATITUDE']
    longitude = row['LONGITUDE']
    county_centroids_dict[county_fips] = np.array([latitude, longitude])

tract_centroids = pd.read_csv('https://www2.census.gov/geo/docs/reference/cenpop2020/tract/CenPop2020_Mean_TR.txt')
tract_centroids['STATEFP'] = tract_centroids['STATEFP'].astype(str).str.zfill(2)
tract_centroids['COUNTYFP'] = tract_centroids['COUNTYFP'].astype(str).str.zfill(3)
tract_centroids['TRACTCE'] = tract_centroids['TRACTCE'].astype(str).str.zfill(6)
tract_centroids['tract_fips'] = tract_centroids['STATEFP'] + tract_centroids['COUNTYFP'] + tract_centroids['TRACTCE']
tract_centroids.set_index('tract_fips', inplace=True)


tract_median_df.set_index('tract_fips', inplace=True)
unique_tracts = set(tract_median_df.index.unique())
progress_bar = tqdm(reachable_tracts.items(), total=len(reachable_tracts), desc='Processing reachable tracts')
for county_fips, neighbor_tracts_set in progress_bar:
    tract_wad_dict = {}
    neighboring_counties_set = set([tract_fips[:5] for tract_fips in neighbor_tracts_set])
    for neighboring_county_fips in neighboring_counties_set:
        tract_wad_filepath = os.path.join(external_ssd_path, f'uniform_pipeline_v3/baseline_conditions_node_level/{neighboring_county_fips}/ct_wad.parquet')
        if os.path.exists(tract_wad_filepath):
            tract_wad_df = pd.read_parquet(tract_wad_filepath)
            tract_wad_df['census_tract_fips'] = tract_wad_df['census_tract_fips'].astype(str).str.zfill(11)
            tract_wad_dict.update(dict(zip(tract_wad_df['census_tract_fips'], tract_wad_df['weighted_avg_distance'])))

    
    if not county_fips in upper_bound_incoming_visits:
        upper_bound_incoming_visits[county_fips] = 0
    neighbor_tracts_set1 = list(set(neighbor_tracts_set).intersection(unique_tracts).intersection(tract_centroids.index.unique()))
    
    curr_county_centroid = county_centroids_dict[county_fips]
    curr_tract_centroids = tract_centroids.loc[neighbor_tracts_set1]
    curr_tract_centroids_coords = [np.array([row['LATITUDE'], row['LONGITUDE']]) for _, row in curr_tract_centroids.iterrows()]
    curr_tract_centroids_indices = [idx for idx in curr_tract_centroids.index]
    if len(curr_tract_centroids_indices) == 0:
        continue
    distances = haversine_vec(curr_county_centroid, curr_tract_centroids_coords)
    curr_tract_distances = dict(zip(curr_tract_centroids_indices, distances))
    
    
    curr_median_df = tract_median_df.loc[neighbor_tracts_set1]
    curr_median_df_nan_visits = curr_median_df.loc[curr_median_df.median_visits.isna()]
    curr_median_df_nan_visits['tract_wad'] = curr_median_df_nan_visits.index.map(tract_wad_dict).fillna(0.0)
    assert curr_median_df_nan_visits.tract_wad.sum() == 0.0, f'Non zero WAD'
    # Actually all the NaN tracts are assigned WAD = 0.0
    curr_median_df_nan_visits['tract_distance'] = curr_median_df_nan_visits.index.map(curr_tract_distances).fillna(0.0)
    curr_median_df_nan_visits['pct_cross_county'] = curr_median_df_nan_visits.index.str[:5].map(cross_county_percentage_by_county)
    nan_mask = curr_median_df_nan_visits['pct_cross_county'].isna()
    if nan_mask.any():
        curr_median_df_nan_visits.loc[nan_mask, 'pct_cross_county'] = (
            curr_median_df_nan_visits.index[nan_mask].str[:2].map(cross_county_percentage_by_state)
        )
    curr_median_df_nan_visits['pct_cross_county'] = curr_median_df_nan_visits['pct_cross_county'].fillna(0.0)
    curr_median_df_nan_visits['out_of_county_visits'] = curr_median_df_nan_visits['esrd_patients'] * curr_median_df_nan_visits['pct_cross_county'] / 100.0
    upper_bound_incoming_visits[county_fips] += curr_median_df_nan_visits[curr_median_df_nan_visits.tract_wad < curr_median_df_nan_visits.tract_distance]['out_of_county_visits'].sum()


lower_bound_incoming_visits = pd.DataFrame(lower_bound_incoming_visits.items(), columns=['facility_county_fips', 'adjusted_median_visits'])
upper_bound_incoming_visits = pd.DataFrame(upper_bound_incoming_visits.items(), columns=['facility_county_fips', 'adjusted_median_visits'])


lower_bound_incoming_visits = lower_bound_incoming_visits.merge(county_working_hours, on='facility_county_fips')
upper_bound_incoming_visits = upper_bound_incoming_visits.merge(county_working_hours, on='facility_county_fips')
upper_bound_incoming_visits['dialysis_burden'] = upper_bound_incoming_visits['adjusted_median_visits']*3 / upper_bound_incoming_visits['max_sessions']
upper_bound_incoming_visits['dialysis_burden'] = upper_bound_incoming_visits['dialysis_burden'].round(3)
upper_bound_incoming_visits['dialysis_burden_urban'] = upper_bound_incoming_visits['adjusted_median_visits']*3 / upper_bound_incoming_visits['max_sessions_urban']
upper_bound_incoming_visits['dialysis_burden_urban'] = upper_bound_incoming_visits['dialysis_burden_urban'].round(3)
upper_bound_incoming_visits['dialysis_burden_rural'] = upper_bound_incoming_visits['adjusted_median_visits']*3 / upper_bound_incoming_visits['max_sessions_rural']
upper_bound_incoming_visits['dialysis_burden_rural'] = upper_bound_incoming_visits['dialysis_burden_rural'].round(3)
lower_bound_incoming_visits['dialysis_burden'] = lower_bound_incoming_visits['adjusted_median_visits']*3 / lower_bound_incoming_visits['max_sessions']
lower_bound_incoming_visits['dialysis_burden'] = lower_bound_incoming_visits['dialysis_burden'].round(3)
lower_bound_incoming_visits['dialysis_burden_urban'] = lower_bound_incoming_visits['adjusted_median_visits']*3 / lower_bound_incoming_visits['max_sessions_urban']
lower_bound_incoming_visits['dialysis_burden_urban'] = lower_bound_incoming_visits['dialysis_burden_urban'].round(3)
lower_bound_incoming_visits['dialysis_burden_rural'] = lower_bound_incoming_visits['adjusted_median_visits']*3 / lower_bound_incoming_visits['max_sessions_rural']
lower_bound_incoming_visits['dialysis_burden_rural'] = lower_bound_incoming_visits['dialysis_burden_rural'].round(3)



lower_bound_incoming_visits.drop(columns=['adjusted_median_visits', 'max_sessions', 'max_sessions_urban', 'max_sessions_rural'], inplace=True)
upper_bound_incoming_visits.drop(columns=['adjusted_median_visits', 'max_sessions', 'max_sessions_urban', 'max_sessions_rural'], inplace=True)

incoming_visits = lower_bound_incoming_visits.merge(upper_bound_incoming_visits, on='facility_county_fips', how='right', suffixes=('_LB', '_UB'))
print(f'incoming_visits: \n{incoming_visits.head()}')
incoming_visits['dialysis_burden_LB'].fillna(0.0, inplace=True)
incoming_visits['dialysis_burden_urban_LB'].fillna(0.0, inplace=True)
incoming_visits['dialysis_burden_urban_UB'].fillna(0.0, inplace=True)
incoming_visits['dialysis_burden_rural_LB'].fillna(0.0, inplace=True)
incoming_visits['dialysis_burden_rural_UB'].fillna(0.0, inplace=True)
incoming_visits['dialysis_burden_Crude'] = (incoming_visits['dialysis_burden_LB'] + incoming_visits['dialysis_burden_UB']) / 2.0
incoming_visits['dialysis_burden_urban_Crude'] = (incoming_visits['dialysis_burden_urban_LB'] + incoming_visits['dialysis_burden_urban_UB']) / 2.0
incoming_visits['dialysis_burden_rural_Crude'] = (incoming_visits['dialysis_burden_rural_LB'] + incoming_visits['dialysis_burden_rural_UB']) / 2.0
incoming_visits['dialysis_burden_pct_rank'] = incoming_visits['dialysis_burden_Crude'].rank(pct=True)
incoming_visits['dialysis_burden_urban_pct_rank'] = incoming_visits['dialysis_burden_urban_Crude'].rank(pct=True)
incoming_visits['dialysis_burden_rural_pct_rank'] = incoming_visits['dialysis_burden_rural_Crude'].rank(pct=True)
print(f'incoming_visits description: \n{incoming_visits.describe()}')

incoming_visits.to_csv(os.path.join(dialysis_burden_access_dirpath, f'dialysis_burden_all_facilities_range_{cutoff}miles.csv'), index=False)
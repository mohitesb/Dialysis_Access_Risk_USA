import pandas as pd
import geopandas as gpd
import os
import utils
import numpy as np
from tqdm.auto import tqdm
from dataloader import SafeGraph
from utils import get_good_cms_and_walking_patterns
from dataloader import NeighborhoodExtraction
from utils import haversine_vec
import random
import copy
import pickle

geographic_crs = 'EPSG:4326'
external_ssd_path = os.path.join(utils.EXTERNAL_SSD_ROOT, 'Data/')
dialysis_burden_access_dirpath = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/dialysis_burden_access_2023')
cms_dialysis_filepath = os.path.join(utils.ESRD_ROOT, 'Data/CMS_Dialysis_Locations/DFC_FACILITY_arcgis_2024.geojson')
svi_df_filepath = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/SVI_2020_US_county/SVI2020_US_county.gdb')
if not dialysis_burden_access_dirpath:
    os.makedirs(dialysis_burden_access_dirpath)

svi_df = gpd.read_file(svi_df_filepath)
if svi_df.crs is None:
    svi_df = svi_df.to_crs(geographic_crs)
else:
    svi_df.to_crs(geographic_crs, inplace=True)
svi_df['FIPS'] = svi_df['FIPS'].astype(str).str.zfill(5)

# Add th
cms_df = gpd.read_file(cms_dialysis_filepath)
if cms_df.crs is None:
    cms_df = cms_df.to_crs(geographic_crs)
else:
    cms_df.to_crs(geographic_crs, inplace=True)
cms_df['CMS Certification Number (CCN)'] = cms_df['CMS Certification Number (CCN)'].astype(int)

cms_df = gpd.sjoin(cms_df, svi_df[['FIPS', 'geometry']], predicate='within')
cms_df.rename(columns={'FIPS': 'facility_county_fips'}, inplace=True)


esrd_patients_filepath = os.path.join(utils.ESRD_ROOT, 'Data/esrdCountyLevelData/ESRD_Hemo_2023/esrd_patients.pkl')
if os.path.exists(esrd_patients_filepath):
    with open(esrd_patients_filepath, 'rb') as f:
        esrd_patients = pickle.load(f)
else:
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

# cb_census_dirpath = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/census_block_populations')
# county_pops = {}
# tract_pops = []
# for filename in os.listdir(cb_census_dirpath):
#     if not filename.endswith('.csv'):
#         continue
#     county_fips = filename[:5]
#     cb_census_filepath = os.path.join(cb_census_dirpath, filename)
#     temp_df = pd.read_csv(cb_census_filepath)
#     temp_df['state'] = temp_df.state.astype(str).str.zfill(2)
#     temp_df['county'] = temp_df.county.astype(str).str.zfill(3)
#     temp_df['tract'] = temp_df.tract.astype(str).str.zfill(6)
#     temp_df['block'] = temp_df.block.astype(str).str.zfill(4)
#     temp_df['cb_fips'] = temp_df.state + temp_df.county + temp_df.tract + temp_df.block
#     county_pops[county_fips] = temp_df.P1_001N.sum()

#     temp_df['tract_fips'] = temp_df['cb_fips'].str[:11]
#     curr_tract_pops = temp_df[['tract_fips', 'P1_001N']].groupby('tract_fips').sum().reset_index()
#     curr_tract_pops.rename(columns={'P1_001N': 'population'}, inplace=True)
#     tract_pops.append(curr_tract_pops)
# tract_pops = pd.concat(tract_pops)


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

esrd_rates_filepath = os.path.join(utils.ESRD_ROOT, 'Data/esrdCountyLevelData/ESRD_Hemo_2023/esrd_rates.pkl')
if os.path.exists(esrd_rates_filepath):
    with open(esrd_rates_filepath, 'rb') as f:
        esrd_rates = pickle.load(f)
else:
    esrd_rates = {county_fips:esrd_pop/county_pops[county_fips] for county_fips, esrd_pop in esrd_patients.items() if county_fips in county_pops and county_pops[county_fips] > 0}
    
tract_pops['esrd_rate'] = tract_pops.tract_fips.str[:5].map(esrd_rates).fillna(0)
tract_pops['esrd_patients'] = tract_pops.population * tract_pops.esrd_rate
tract_pops = tract_pops.loc[tract_pops.esrd_patients > 0]



svi_df['GEOID'] = svi_df['FIPS'].astype(str).str.zfill(5)
tract_svi_filepath = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/SVI_2020_US/SVI2020_US_tract.gdb')
tract_svi_df = gpd.read_file(tract_svi_filepath)
if tract_svi_df.crs is None:
    tract_svi_df = tract_svi_df.to_crs(geographic_crs)
else:
    tract_svi_df.to_crs(geographic_crs, inplace=True)
tract_svi_df['GEOID'] = tract_svi_df['FIPS'].astype(str).str.zfill(11)


cutoff = 100.0
od_median_dirpath = os.path.join(external_ssd_path, 'uniform_pipeline_v3/baseline_conditions_node_level')
od_median_df = []
for county_fips, county_geometry in tqdm(zip(svi_df.FIPS, svi_df.geometry), total=len(svi_df), desc='Processing counties'):
    od_median_filepath = os.path.join(od_median_dirpath, f'{county_fips}/od_median.csv')
    if not os.path.exists(od_median_filepath):
        continue
    try:
        curr_od_median_df = pd.read_csv(od_median_filepath)
    except:
        print(f'Error reading od_median_filepath: {od_median_filepath}')
        continue
    od_median_df.append(curr_od_median_df)
    # if len(od_median_df) > 40:
    #     break

# good_cms_df = cms_df.copy()
good_cms_df = copy.deepcopy(cms_df)
# good_cms_df = gpd.sjoin(good_cms_df, svi_df[['FIPS', 'geometry']], predicate='within')
good_cms_df.rename(columns={'FIPS': 'facility_county_fips'}, inplace=True)


od_median_df = pd.concat(od_median_df)
od_median_df['cbg_geoid'] = od_median_df['cbg_geoid'].astype(str).str.zfill(12)
od_median_df['cms_id'] = od_median_df['cms_id'].astype(int)
od_median_df['county_fips'] = od_median_df['cbg_geoid'].str[:5]
od_median_df = od_median_df.loc[od_median_df.median_visits > 0]
od_median_df = od_median_df.loc[od_median_df.median_visits.notna()]
od_median_df['tract_fips'] = od_median_df.cbg_geoid.str[:11]
od_median_df.drop(columns=['cbg_geoid'], inplace=True)
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
cross_county_percentage_by_state.sort_values(ascending=False, inplace=True)
print(f'cross_county_percentage_by_state: \n{cross_county_percentage_by_state}')
cross_county_percentage_by_state = cross_county_percentage_by_state.to_dict()

cross_counts_county = cross_county.groupby('county_fips')['adjusted_median_visits'].sum()
within_counts_county = within_county.groupby('county_fips')['adjusted_median_visits'].sum()
total_counts_county = valid.groupby('county_fips')['adjusted_median_visits'].sum()
cross_county_percentage_by_county = (cross_counts_county / total_counts_county * 100).fillna(0)
cross_county_percentage_by_county.sort_values(ascending=False, inplace=True)
print(f'cross_county_percentage_by_county: \n{cross_county_percentage_by_county}')
cross_county_percentage_by_county = cross_county_percentage_by_county.to_dict()

print(f'tract_median_df: \n{tract_median_df.head()}')



##########################################################################################ESTIMATE DIALYSIS ACCESS RANGE############################################################################################################
dialysis_burden_access_dirpath = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/dialysis_burden_access_2023')
dialysis_burden = pd.read_csv(os.path.join(dialysis_burden_access_dirpath, f'dialysis_burden_all_facilities_range_{cutoff}miles.csv'))
dialysis_burden['facility_county_fips'] = dialysis_burden['facility_county_fips'].astype(str).str.zfill(5)
if not 'dialysis_burden_Crude' in dialysis_burden.columns:
    dialysis_burden['dialysis_burden_Crude'] = (dialysis_burden['dialysis_burden_LB'] + dialysis_burden['dialysis_burden_UB']) / 2.0
dialysis_burden = dialysis_burden.loc[dialysis_burden.dialysis_burden_Crude.notna()]
print(f'dialysis_burden: \n{dialysis_burden.head()}')


dialysis_burden = dict(zip(
    dialysis_burden['facility_county_fips'],
    zip(dialysis_burden['dialysis_burden_LB'], dialysis_burden['dialysis_burden_UB'], dialysis_burden['dialysis_burden_Crude'])
))


county_centroids = pd.read_csv('https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt')
county_centroids['STATEFP'] = county_centroids['STATEFP'].astype(str).str.zfill(2)
county_centroids['COUNTYFP'] = county_centroids['COUNTYFP'].astype(str).str.zfill(3)
county_centroids['county_fips'] = county_centroids['STATEFP'] + county_centroids['COUNTYFP']
county_centroids['LATITUDE'] = county_centroids['LATITUDE'].astype(float)
county_centroids['LONGITUDE'] = county_centroids['LONGITUDE'].astype(float)
county_centroids.set_index('county_fips', inplace=True)

tract_centroids_dict = {}
tract_centroids = pd.read_csv('https://www2.census.gov/geo/docs/reference/cenpop2020/tract/CenPop2020_Mean_TR.txt')
tract_centroids['STATEFP'] = tract_centroids['STATEFP'].astype(str).str.zfill(2)
tract_centroids['COUNTYFP'] = tract_centroids['COUNTYFP'].astype(str).str.zfill(3)
tract_centroids['TRACTCE'] = tract_centroids['TRACTCE'].astype(str).str.zfill(6)
tract_centroids['tract_fips'] = tract_centroids['STATEFP'] + tract_centroids['COUNTYFP'] + tract_centroids['TRACTCE']
# tract_centroids.set_index('tract_fips', inplace=True)
for _, row in tract_centroids.iterrows():
    tract_fips = row['tract_fips']
    latitude = row['LATITUDE']
    longitude = row['LONGITUDE']
    tract_centroids_dict[tract_fips] = np.array([latitude, longitude])



lower_bounds = {}

for _, row in tract_median_df[tract_median_df.median_visits.notna()].iterrows():
    tract_fips = row['tract_fips']
    county_fips = tract_fips[:5]
    if not county_fips in lower_bounds:
        lower_bounds[county_fips] = {
            'weights': [],
            'vals': []
        }
    
    facility_county_fips = row['facility_county_fips']
    lower_bounds[county_fips]['weights'].append(row['adjusted_median_visits'])
    if facility_county_fips in dialysis_burden:
        vals = np.array(list(dialysis_burden[facility_county_fips]))
    else:
        vals = np.zeros(3)
    lower_bounds[county_fips]['vals'].append(vals)

svi_df['GEOID'] = svi_df['FIPS'].astype(str).str.zfill(5)
reachable_counties = {}
reachable_counties_filepath = os.path.join(dialysis_burden_access_dirpath, f'reachable_counties_{cutoff}miles.pkl')
if os.path.exists(reachable_counties_filepath):
    with open(reachable_counties_filepath, 'rb') as f:
        reachable_counties = pickle.load(f)
else:
    for tract_fips, tract_geometry in tqdm(zip(tract_svi_df.GEOID, tract_svi_df.geometry), total=len(tract_svi_df), desc='Processing tracts'):
        ne = NeighborhoodExtraction(svi_df, tract_geometry, cutoff=cutoff)
        study_area_geoids = ne.extract_study_area()
        reachable_counties[tract_fips] = study_area_geoids
        # if len(reachable_counties) > 200:
        #     break
    with open(reachable_counties_filepath, 'wb') as f:
        pickle.dump(reachable_counties, f)

print(f'reachable_counties: \n{list(reachable_counties.items())[:10]}')

progress_bar = tqdm(reachable_counties.items(), total=len(reachable_counties), desc='Processing reachable counties')
county_tracts = {}
for tract_fips in reachable_counties.keys():
    county_fips = tract_fips[:5]
    if county_fips not in county_tracts:
        county_tracts[county_fips] = set()
    county_tracts[county_fips].add(tract_fips)

# upper_bounds = lower_bounds.copy()
upper_bounds = copy.deepcopy(lower_bounds)
unique_county_fips = set(list(dialysis_burden.keys()))
print(f'unique_county_fips: \n{len(unique_county_fips)}')
print(list(unique_county_fips)[:10])
print(f'county_centroids: \n{county_centroids.head()}')
tract_median_df.set_index('tract_fips', inplace=True)
progress_bar = tqdm(enumerate(county_tracts.items()), total=len(county_tracts), desc='Processing county tracts')
for i, (county_fips, tract_fips_set) in progress_bar:
    tract_wad_filepath = os.path.join(external_ssd_path, f'uniform_pipeline_v3/baseline_conditions_node_level/{county_fips}/ct_wad.parquet')
    if not os.path.exists(tract_wad_filepath):
        # print(f'tract_wad_filepath not found at: {tract_wad_filepath}')
        continue
    tract_wad_df = pd.read_parquet(tract_wad_filepath)
    tract_wad_df['census_tract_fips'] = tract_wad_df['census_tract_fips'].astype(str).str.zfill(11)
    tract_wad_dict = dict(zip(tract_wad_df['census_tract_fips'], tract_wad_df['weighted_avg_distance']))
    for tract_fips in tract_fips_set:
        neighbor_counties_set = reachable_counties[tract_fips]
        # print(f'neighbor_counties_set before intersection: \n{len(set(neighbor_counties_set))}')
        neighbor_counties_set1 = list(set(neighbor_counties_set).intersection(set(list(county_centroids.index.unique()))).intersection(unique_county_fips))
        if len(neighbor_counties_set1) == 0:
            # print(f'No neighbor counties found for {tract_fips}')
            continue
        # print(f'neighbor_counties_set1 after intersection: \n{len(set(neighbor_counties_set1))}')
        # curr_tract_centroid = tract_centroids_dict[tract_fips]
        curr_county_centroids = county_centroids.loc[neighbor_counties_set1]
        
        curr_county_centroids_indices = [idx for idx in curr_county_centroids.index]
        # curr_county_centroids_coords = [np.array([row['LATITUDE'], row['LONGITUDE']]) for _, row in curr_county_centroids.iterrows()]
        # distances = haversine_vec(curr_tract_centroid, curr_county_centroids_coords)
        # curr_county_distances = dict(zip(curr_county_centroids_indices, distances))
        curr_county_burden = np.array([dialysis_burden[county_fips1][2] for county_fips1 in curr_county_centroids_indices])
        curr_county_burden_lb = np.array([dialysis_burden[county_fips1][0] for county_fips1 in curr_county_centroids_indices])
        curr_county_burden_ub = np.array([dialysis_burden[county_fips1][1] for county_fips1 in curr_county_centroids_indices])
        
        # Find the county with min burden among neighboring counties
        visiting_county_fips_lb = curr_county_centroids_indices[np.argmin(curr_county_burden)]
        visiting_county_fips_ub = curr_county_centroids_indices[np.argmax(curr_county_burden)]
        # print(f'visiting_county_fips_lb: {visiting_county_fips_lb}, {dialysis_burden[visiting_county_fips_lb]}')
        # print(f'visiting_county_fips_ub: {visiting_county_fips_ub}, {dialysis_burden[visiting_county_fips_ub]}')
        
        if not tract_fips in tract_median_df.index.unique():
            # print(f'Tract {tract_fips} not found in tract_median_df. Skipping...')
            continue

        curr_median_df = tract_median_df.loc[[tract_fips]]
        curr_median_df_nan_visits = curr_median_df.loc[curr_median_df.median_visits.isna()]
        if len(curr_median_df_nan_visits) == 0:
            # print(f'No nan visits found for tract {tract_fips}. Skipping...')
            continue

        curr_median_df_nan_visits['tract_wad'] = curr_median_df_nan_visits.index.map(tract_wad_dict).fillna(0.0)
        assert curr_median_df_nan_visits['tract_wad'].sum() == 0.0, f'Non zero WAD != 0.0'
        # curr_median_df_nan_visits['county_distance'] = curr_median_df_nan_visits.index.map(curr_county_distances).fillna(0.0)
        # curr_median_df_nan_visits['pct_cross_county'] = curr_median_df_nan_visits.index.str[:2].map(cross_county_percentage_by_state).fillna(0.0)
        
        curr_median_df_nan_visits['pct_cross_county'] = curr_median_df_nan_visits.index.str[:5].map(cross_county_percentage_by_county)
        nan_mask = curr_median_df_nan_visits['pct_cross_county'].isna()
        if nan_mask.any():
            curr_median_df_nan_visits.loc[nan_mask, 'pct_cross_county'] = (
                curr_median_df_nan_visits.index[nan_mask].str[:2].map(cross_county_percentage_by_state)
            )
        curr_median_df_nan_visits['out_of_county_visits'] = curr_median_df_nan_visits['esrd_patients'] * curr_median_df_nan_visits['pct_cross_county'] / 100.0
        curr_median_df_nan_visits['in_county_visits'] = curr_median_df_nan_visits['esrd_patients'] - curr_median_df_nan_visits['out_of_county_visits']
        
        # print(f'curr_median_df_nan_visits: \n{curr_median_df_nan_visits.head()}')

        for tract_fips1, out_of_county_visits, in_county_visits in zip(curr_median_df_nan_visits.index, curr_median_df_nan_visits['out_of_county_visits'], curr_median_df_nan_visits['in_county_visits']):
            county_fips1 = tract_fips1[:5]
            if not county_fips1 in lower_bounds:
                lower_bounds[county_fips1] = {
                    'weights': [],
                    'vals': []
                }
            if not county_fips1 in upper_bounds:
                upper_bounds[county_fips1] = {
                    'weights': [],
                    'vals': []
                }
            # In county visits
            upper_bounds[county_fips1]['weights'].append(in_county_visits)
            if county_fips1 in dialysis_burden:
                upper_bounds[county_fips1]['vals'].append(
                    np.array(list(dialysis_burden[county_fips1]))
                )
            else:
                upper_bounds[county_fips1]['vals'].append(np.array([0.0, 0.0, 0.0]))
            
            # Out of state visits max burden
            upper_bounds[county_fips1]['weights'].append(out_of_county_visits)
            upper_bounds[county_fips1]['vals'].append(
                    np.array(list(dialysis_burden[visiting_county_fips_ub]))
                )
                

            
            # In county visits
            lower_bounds[county_fips1]['weights'].append(in_county_visits)
            if county_fips1 in dialysis_burden:
                lower_bounds[county_fips1]['vals'].append(
                    np.array(list(dialysis_burden[county_fips1]))
                )
            else:
                lower_bounds[county_fips1]['vals'].append(np.array([0.0, 0.0, 0.0]))
            
            # Out of state visits min burden
            lower_bounds[county_fips1]['weights'].append(out_of_county_visits)
            lower_bounds[county_fips1]['vals'].append(
                np.array(list(dialysis_burden[visiting_county_fips_lb]))
            )

print(f'Checking if Dialysis Access Strain LB & UB are unique')
flag = False
for k, v in lower_bounds.items():
    if k in upper_bounds:
        if v['vals'][-1][-1] != upper_bounds[k]['vals'][-1][-1]:
            flag = True
            print(k, v['vals'][-1], upper_bounds[k]['vals'][-1])
            break
assert flag, 'Dialysis Access Strain LB & UB are never unique'
# exit()



def weighted_average(weights, vals):
    # return np.sum(weights*vals, axis=0) / np.sum(weights)
    return np.average(vals, weights=weights, axis=0)

dialysis_access_lb = {k : weighted_average(v['weights'], v['vals']) for k, v in lower_bounds.items()}
dialysis_access_ub = {k : weighted_average(v['weights'], v['vals']) for k, v in upper_bounds.items()}

# print(f'Dialysis Access Strain LB & UB')
# for i, (k, v) in enumerate(dialysis_access_lb.items()):
#     if not k.startswith('01'):
#         continue
#     print(k, v, dialysis_access_ub[k])
#     if i > 20:
#         exit()

dialysis_access_df = pd.DataFrame(
    [(county_fips, dialysis_access_lb[county_fips][-1], dialysis_access_ub[county_fips][-1]) for county_fips in set(list(dialysis_access_lb.keys())).union(set(list(dialysis_access_ub.keys())))],
    columns=['county_fips', 'dialysis_access_LB', 'dialysis_access_UB']
)
dialysis_access_df['dialysis_access_Crude'] = (dialysis_access_df['dialysis_access_LB'] + dialysis_access_df['dialysis_access_UB']) / 2.0
dialysis_access_df['dialysis_access_pct_rank'] = dialysis_access_df['dialysis_access_Crude'].rank(pct=True)
dialysis_access_df.to_csv(os.path.join(dialysis_burden_access_dirpath, f'dialysis_access_all_facilities_range_{cutoff}miles.csv'), index=False)

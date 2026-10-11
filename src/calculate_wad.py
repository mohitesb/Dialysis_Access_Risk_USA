import pandas as pd
import geopandas as gpd
import os
import utils
import pickle
import numpy as np
import logging
from tqdm.auto import tqdm

logging.basicConfig(
    filename='calculate_wad_all_cms.log',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)



geographic_crs = 'EPSG:4326'
external_ssd_path = os.path.join(utils.EXTERNAL_SSD_ROOT, 'Data')
od_median_filepath_template = os.path.join(external_ssd_path, 'uniform_pipeline_v3/baseline_conditions_node_level/{}/od_median.csv')
od_median_filepath_template = os.path.join(external_ssd_path, 'uniform_pipeline_v3/baseline_conditions_node_level/{}/od_median.csv')
od_dist_filepath_template = os.path.join(external_ssd_path, 'uniform_pipeline_v3/baseline_conditions_node_level/{}/od_dist_all_cms.parquet')
cms_id_map_filepath_template = os.path.join(external_ssd_path, 'uniform_pipeline_v3/baseline_conditions_node_level/{}/local_cms_id_map_all_cms.pkl')
road_nodes_map_filepath_template = os.path.join(external_ssd_path, 'uniform_pipeline_v3/baseline_conditions_node_level/{}/local_node_id_map_all_cms.pkl')
# cb_road_nodes_population_filepath_template = os.path.join(external_ssd_path, 'uniform_pipeline_v3/Roads/cb_road_nodes_populations/{}.csv')
cb_road_nodes_population_filepath_template = os.path.join(utils.ESRD_ROOT, "Data/uniform_pipeline_v3/baseline_conditions_node_level_2023/{}/cb_road_nodes_populations.csv")


urban_data_filepath = os.path.join(utils.ESRD_ROOT, 'Data/Urban/tl_2020_us_uac20.shp')
urban_df = gpd.read_file(urban_data_filepath)
if urban_df.crs is None:
    urban_df.crs = geographic_crs
else:
    urban_df.to_crs(geographic_crs, inplace=True)


counties_filepath = os.path.join(utils.ESRD_ROOT, 'Data/County_Shapefiles/tl_2020_us_county.shp')
counties = gpd.read_file(counties_filepath)
counties['GEOID'] = counties.GEOID.astype(str).str.zfill(5)
counties.crs = geographic_crs
counties = counties.sort_values(by=['GEOID'])
states = gpd.read_file(os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/state_boundaries/tl_2024_us_state.shp'))
states['STATEFP'] = states.STATEFP.astype(str).str.zfill(2)
counties['STATEFP'] = counties.STATEFP.astype(str).str.zfill(2)
states.rename(columns={'NAME': 'STNAME'}, inplace=True)
counties = counties.merge(states[['STATEFP', 'STUSPS', 'STNAME']], on='STATEFP')
# coastal_state_abbs = [
#     'ME', 'NH', 'MA', 'RI', 'CT',  # New England Atlantic
#     'NY', 'NJ', 'DE', 'MD', 'VA', 'NC', 'SC', 'GA', 'FL',  # Middle/South Atlantic/Gulf
#     'AL', 'MS', 'LA', 'TX',  # Gulf
#     'CA', 'OR', 'WA',  # Pacific
#     'AK', 'HI'  # Non-Contiguous
# ]
local_path = utils.ESRD_ROOT
# wad_filepath_template = os.path.join(external_ssd_path, 'uniform_pipeline_v3/baseline_conditions_node_level/{}/cbg_wad_all_cms.parquet')
# ct_wad_filepath_template = os.path.join(external_ssd_path, 'uniform_pipeline_v3/baseline_conditions_node_level/{}/ct_wad_all_cms.parquet')
# county_wad_filepath_template = os.path.join(external_ssd_path, 'uniform_pipeline_v3/baseline_conditions_node_level/{}/county_wad_all_cms.parquet')
wad_filepath_template = os.path.join(local_path, 'Data/uniform_pipeline_v3/baseline_conditions_node_level_2023/{}/cbg_wad_all_cms.csv')
ct_wad_filepath_template = os.path.join(local_path, 'Data/uniform_pipeline_v3/baseline_conditions_node_level_2023/{}/ct_wad_all_cms.csv')
county_wad_filepath_template = os.path.join(local_path, 'Data/uniform_pipeline_v3/baseline_conditions_node_level_2023/{}/county_wad_all_cms.csv')
cb_wad_filepath_template = os.path.join(local_path, 'Data/uniform_pipeline_v3/baseline_conditions_node_level_2023/{}/cb_wad_all_cms.csv')
urban_rural_cb_wad_filepath_template = os.path.join(local_path, 'Data/uniform_pipeline_v3/baseline_conditions_node_level_2023/{}/urban_rural_cb_wad_all_cms.csv')
urban_rural_cbg_wad_filepath_template = os.path.join(local_path, 'Data/uniform_pipeline_v3/baseline_conditions_node_level_2023/{}/urban_rural_cbg_wad_all_cms.csv')
urban_rural_ct_wad_filepath_template = os.path.join(local_path, 'Data/uniform_pipeline_v3/baseline_conditions_node_level_2023/{}/urban_rural_ct_wad_all_cms.csv')
urban_rural_county_wad_filepath_template = os.path.join(local_path, 'Data/uniform_pipeline_v3/baseline_conditions_node_level_2023/{}/urban_rural_county_wad_all_cms.csv')
wad_dirpath_template = os.path.join(local_path, 'Data/uniform_pipeline_v3/baseline_conditions_node_level_2023/{}')
road_nodes_filepath_template = os.path.join(utils.EXTERNAL_SSD_ROOT, 'Data/uniform_pipeline_v3/Roads/Nodes/nodes_{}.shp')
road_nodes_dirpath = os.path.join(utils.EXTERNAL_SSD_ROOT, "Data/uniform_pipeline_v3/Roads/Nodes")
import glob

def find_node_files(base_dir, county_geoid):
    """
    Finds node shapefiles in the given directory according to two patterns:
    1. files like nodes_10001.shp directly under base_dir
    2. files under subdirectories matching FILE*.CHK/nodes_*.shp
    Returns a list of found file paths.
    """
    pattern1 = os.path.join(base_dir, f"nodes_{county_geoid}.shp")
    pattern2 = os.path.join(base_dir, f"FILE*.CHK", f"nodes_{county_geoid}.shp.shp")
    files1 = glob.glob(pattern1)
    files2 = glob.glob(pattern2)
    return files1 + files2
# print(find_node_files(road_nodes_dirpath, '01001'))
# print(find_node_files(road_nodes_dirpath, '01007'))
# exit()


cb_census_dirpath = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/census_block_populations')
county_pops = {}
for filename in tqdm(list(os.listdir(cb_census_dirpath))):
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
    temp_df['county_fips'] = temp_df.cb_fips.str[:5]

    duplicate_mask = temp_df.duplicated(subset=['cb_fips'], keep=False)
    duplicate_pairs = temp_df.loc[duplicate_mask, ['cb_fips']]
    duplicate_pairs.sort_values(by=['cb_fips'], inplace=True)
    if not duplicate_pairs.empty:
        print(f"Duplicate cb_fips found in {filename}:")
        print(duplicate_pairs.value_counts().reset_index(name='count'))
        print(temp_df.loc[temp_df.cb_fips.isin(duplicate_pairs.cb_fips)][['cb_fips', 'P1_001N']].sort_values(by='cb_fips'))
        exit()

    temp_df.drop(columns=['cb_fips'], inplace=True)
    county_pops[county_fips] = temp_df.P1_001N.sum()
    
esrd_patients_dirpath = os.path.join(utils.ESRD_ROOT, "Data/esrdCountyLevelData/ESRD_Hemo_2023")
esrd_rate_filepath = os.path.join(utils.ESRD_ROOT, 'Data/esrdCountyLevelData/ESRD_Hemo_2023/esrd_rates.pkl')
esrd_patients_filepath = os.path.join(utils.ESRD_ROOT, 'Data/esrdCountyLevelData/ESRD_Hemo_2023/esrd_patients.pkl')
if os.path.exists(esrd_rate_filepath) and os.path.exists(esrd_patients_filepath):
    esrd_rate = pickle.load(open(esrd_rate_filepath, 'rb'))
    esrd_patients = pickle.load(open(esrd_patients_filepath, 'rb'))
else:
    if esrd_patients_dirpath.endswith('ESRD_Hemo_2023'):
        check_str = 'Incident ESRD Count'
    else:
        check_str = 'Prevalent ESRD Hemo'
    esrd_patients = {}
    for fname in tqdm(list(os.listdir(esrd_patients_dirpath))):
        if fname.startswith(check_str) and fname.endswith('.csv'):
            filename = os.path.join(esrd_patients_dirpath, fname)
            df = pd.read_csv(filename, low_memory=False)
            if not 'FIPS' in df.columns:
                continue
            df['FIPS'] = df.FIPS.astype(str).str.zfill(5)
            df['Prevalent Count'] = df['Prevalent Count'].astype(str).str.replace(',', '').str.replace('*', '0').astype(float)
            patients = dict(zip(df.FIPS, df['Prevalent Count']))
            esrd_patients.update(patients)
    esrd_rate = {county_fips:esrd_pop/county_pops[county_fips] for county_fips, esrd_pop in esrd_patients.items() if county_fips in county_pops and county_pops[county_fips] > 0 and esrd_pop > 0}
    with open(esrd_rate_filepath, 'wb') as f:
        pickle.dump(esrd_rate, f)
    with open(esrd_patients_filepath, 'wb') as f:
        pickle.dump(esrd_patients, f)
counties = counties.sort_values(by=['GEOID'])
progress_bar = tqdm(enumerate(zip(counties.GEOID, counties.NAME, counties.STUSPS, counties.STNAME, counties.geometry)), total=counties.shape[0])
for i, (county_geoid, county_name, state_abb, state_name, county_geometry) in progress_bar:
    # if int(county_geoid[:2]) >= 12:
    #     continue
    wad_dirpath = wad_dirpath_template.format(county_geoid)
    os.makedirs(wad_dirpath, exist_ok=True)
    cbg_save_filepath = wad_filepath_template.format(county_geoid)
    ct_save_filepath = ct_wad_filepath_template.format(county_geoid)
    county_save_filepath = county_wad_filepath_template.format(county_geoid)
    cb_save_filepath = cb_wad_filepath_template.format(county_geoid)
    urban_rural_cbg_wad_filepath = urban_rural_cbg_wad_filepath_template.format(county_geoid)
    urban_rural_ct_wad_filepath = urban_rural_ct_wad_filepath_template.format(county_geoid)
    urban_rural_county_wad_filepath = urban_rural_county_wad_filepath_template.format(county_geoid)
    print(f'Processing county: {county_geoid} {county_name} {state_abb} {state_name}')
    
    logging.info(f'Processing county: {county_geoid} {county_name} {state_abb} {state_name}')
    od_median_filepath = od_median_filepath_template.format(county_geoid)
    if not os.path.exists(od_median_filepath):
        logging.info(f'OD Median file not found for county: {county_geoid} {county_name} {state_abb} {state_name}')
        continue
    od_dist_filepath = od_dist_filepath_template.format(county_geoid)
    if not os.path.exists(od_dist_filepath):
        logging.info(f'OD Dist file not found for county: {county_geoid} {county_name} {state_abb} {state_name}')
        continue
    cms_id_map_filepath = cms_id_map_filepath_template.format(county_geoid)
    road_nodes_map_filepath = road_nodes_map_filepath_template.format(county_geoid)
    cb_road_nodes_population_filepath = cb_road_nodes_population_filepath_template.format(county_geoid)

    od_median_df = pd.read_csv(od_median_filepath)
    od_median_df['cms_id'] = od_median_df['cms_id'].astype(int)
    od_median_df['cbg_geoid'] = od_median_df['cbg_geoid'].astype(str).str.zfill(12)

    od_dist_df = pd.read_parquet(od_dist_filepath)
    cms_id_map = pickle.load(open(cms_id_map_filepath, 'rb'))
    road_nodes_map = pickle.load(open(road_nodes_map_filepath, 'rb'))
    if not os.path.exists(cb_road_nodes_population_filepath):
        logging.info(f'CB Road Nodes Population file not found for county: {county_geoid} {county_name} {state_abb} {state_name}')
        continue
    

    cb_road_nodes_population_df = pd.read_csv(cb_road_nodes_population_filepath)
    cb_road_nodes_population_df['GEOID20'] = cb_road_nodes_population_df['GEOID20'].astype(str).str.zfill(15)
    cb_road_nodes_population_df['cbg_geoid'] = cb_road_nodes_population_df['GEOID20'].str[:12]
    cb_road_nodes_population_df['county_fips'] = cb_road_nodes_population_df['GEOID20'].str[:5]
    duplicate_mask = cb_road_nodes_population_df.duplicated(subset=['osmid'], keep=False)
    duplicate_pairs = cb_road_nodes_population_df.loc[duplicate_mask, ['osmid']]
    duplicate_pairs.sort_values(by=['osmid'], inplace=True)
    if not duplicate_pairs.empty:
        print(f"Duplicate osmid found in cb_road_nodes_population_df: {county_geoid} {county_name} {state_abb} {state_name}:")
        print(duplicate_pairs.value_counts().reset_index(name='count'))
        print(f"{len(duplicate_pairs)} / {cb_road_nodes_population_df.osmid.nunique()}")
        print(cb_road_nodes_population_df.loc[cb_road_nodes_population_df.osmid.isin(duplicate_pairs.osmid)][['osmid', 'GEOID20', 'P1_001N']].sort_values(by='osmid'))
        exit()

    od_dist_df['cms_id'] = od_dist_df['cms_id'].map(cms_id_map).astype(int)
    od_dist_df['osmid'] = od_dist_df['osmid'].map(road_nodes_map).astype(int)
    # Find duplicate (cms_id, osmid) pairs in od_dist_df
    duplicate_mask = od_dist_df.duplicated(subset=['cms_id', 'osmid'], keep=False)
    duplicate_pairs = od_dist_df.loc[duplicate_mask, ['cms_id', 'osmid']]
    duplicate_pairs.sort_values(by=['cms_id', 'osmid'], inplace=True)
    if not duplicate_pairs.empty:
        print(f"Duplicate (cms_id, osmid) pairs found for county: {county_geoid} {county_name} {state_abb} {state_name}:")
        print(duplicate_pairs.value_counts().reset_index(name='count'))

    # cb_road_nodes_population_df['osmid'] = cb_road_nodes_population_df['osmid'].astype(int)
    # cb_road_nodes_population_df = cb_road_nodes_population_df.groupby('osmid').agg({
        
    #     'cbg_geoid': 'first',
    #     'county_fips': 'first',
    #     'P1_001N': 'sum',
    # })
    # Find duplicate (cms_id, osmid) pairs in od_dist_df
    duplicate_mask = cb_road_nodes_population_df.duplicated(subset='osmid', keep=False)
    duplicate_pairs = cb_road_nodes_population_df.loc[duplicate_mask, ['osmid']]
    duplicate_pairs.sort_values(by='osmid', inplace=True)
    if not duplicate_pairs.empty:
        print(f"Duplicate osmid found in cb_road_nodes_population_df: {county_geoid} {county_name} {state_abb} {state_name}:")
        print(duplicate_pairs.value_counts().reset_index(name='count'))
        print(cb_road_nodes_population_df.loc[cb_road_nodes_population_df.osmid.isin(duplicate_pairs.osmid)].sort_values(by='osmid'))

        
    ratio_df = od_median_df.merge(cb_road_nodes_population_df[['osmid', 'GEOID20', 'cbg_geoid', 'county_fips', 'P1_001N', 'P1_001N_ESRD']], on='cbg_geoid', how='inner')
    ratio_df.rename(columns={'P1_001N': 'population', 'P1_001N_ESRD': 'esrd_population'}, inplace=True)
    # Find duplicate (cms_id, osmid) pairs in od_dist_df
    duplicate_mask = ratio_df.duplicated(subset=['cms_id', 'osmid'], keep=False)
    duplicate_pairs = ratio_df.loc[duplicate_mask, ['cms_id', 'osmid']]
    duplicate_pairs.sort_values(by=['cms_id', 'osmid'], inplace=True)
    if not duplicate_pairs.empty:
        print(f"Duplicate (cms_id, osmid) pairs found ratio_df: {county_geoid} {county_name} {state_abb} {state_name}:")
        print(duplicate_pairs.value_counts().reset_index(name='count'))
    
    # ratio_df['esrd_rate'] = ratio_df['county_fips'].map(esrd_rate).fillna(0.0)
    # ratio_df['esrd_population'] = ratio_df['esrd_rate'] * ratio_df['P1_001N']
    # insteaf of commented lines we use precomputed esrd population
    # ratio_df['ratio'] = (
    #     ratio_df['median_visits'] / ratio_df.groupby('osmid')['median_visits'].transform('sum')
    # )
    # ratio_df['esrd_visitors'] = ratio_df['esrd_population'] * ratio_df['ratio']
    # ratio_df['visitors'] = ratio_df['population'] * ratio_df['ratio']
    
    # Find duplicate (cms_id, osmid) pairs in od_dist_df
    duplicate_mask = ratio_df.duplicated(subset=['cms_id', 'osmid'], keep=False)
    duplicate_pairs = ratio_df.loc[duplicate_mask, ['cms_id', 'osmid']]
    duplicate_pairs.sort_values(by=['cms_id', 'osmid'], inplace=True)
    if not duplicate_pairs.empty:
        print(f"Duplicate (cms_id, osmid) pairs found ratio_df after merge with cb_road_nodes_population_df: {county_geoid} {county_name} {state_abb} {state_name}:")
        print(duplicate_pairs.value_counts().reset_index(name='count'))
    
    # Calculate the number of visitors per row
    od_dist_df = od_dist_df.merge(ratio_df, on=['osmid', 'cms_id'])
    # Now compute ratio over only reachable facilities
    od_dist_df['ratio'] = (
        od_dist_df['median_visits'] / od_dist_df.groupby('osmid')['median_visits'].transform('sum')
    )
    od_dist_df['esrd_visitors'] = od_dist_df['esrd_population'] * od_dist_df['ratio']
    od_dist_df['visitors'] = od_dist_df['population'] * od_dist_df['ratio']
    
    od_dist_df['census_tract_fips'] = od_dist_df['cbg_geoid'].str[:11]
    od_dist_df['county_fips'] = od_dist_df['cbg_geoid'].str[:5]

    # Find duplicate (cms_id, osmid) pairs in od_dist_df
    duplicate_mask = od_dist_df.duplicated(subset=['cms_id', 'osmid'], keep=False)
    duplicate_pairs = od_dist_df.loc[duplicate_mask, ['cms_id', 'osmid']]
    duplicate_pairs.sort_values(by=['cms_id', 'osmid'], inplace=True)
    if not duplicate_pairs.empty:
        print(f"After merge with ratio_df, Duplicate (cms_id, osmid) pairs found for county: {county_geoid} {county_name} {state_abb} {state_name}:")
        print(duplicate_pairs.value_counts().reset_index(name='count'))
    
    
    
    
    # road_nodes_filepath = road_nodes_filepath_template.format(county_geoid)
    road_nodes_filepaths = find_node_files(road_nodes_dirpath, county_geoid)
    if len(road_nodes_filepaths) == 0:
        print(f'No road nodes file found for county: {county_geoid} {county_name} {state_abb} {state_name}')
        exit()
    road_nodes_filepath = road_nodes_filepaths[0]
    road_nodes_df = gpd.read_file(road_nodes_filepath)
    road_nodes_df['osmid'] = road_nodes_df['osmid'].astype(int)
    # INSERT_YOUR_CODE
    # Check for duplicate osmids in road_nodes_df
    duplicate_osmids = road_nodes_df[road_nodes_df.duplicated(subset=['osmid'], keep=False)]
    if not duplicate_osmids.empty:
        print(f"Duplicate osmids found in road_nodes_df for county: {county_geoid} {county_name} {state_abb} {state_name}:")
        print(duplicate_osmids[['osmid']].value_counts().reset_index(name='count'))
        exit()

    urban_road_node_osmids = gpd.sjoin(road_nodes_df, urban_df, how='inner', predicate='within')['osmid'].unique()
    od_dist_df['urban'] = od_dist_df['osmid'].isin(urban_road_node_osmids)
    od_dist_df['urban'] = od_dist_df['urban'].astype(bool)

    # Find duplicate (cms_id, osmid) pairs in od_dist_df
    duplicate_mask = od_dist_df.duplicated(subset=['cms_id', 'osmid'], keep=False)
    duplicate_pairs = od_dist_df.loc[duplicate_mask, ['cms_id', 'osmid']]
    duplicate_pairs.sort_values(by=['cms_id', 'osmid'], inplace=True)
    if not duplicate_pairs.empty:
        print(f"After assigning urban, Duplicate (cms_id, osmid) pairs found for county: {county_geoid} {county_name} {state_abb} {state_name}:")
        print(duplicate_pairs.value_counts().reset_index(name='count'))
        exit()
    

    visitors_df = od_dist_df.loc[:, ['cms_id', 'osmid', 'visitors', 'esrd_visitors', 'urban']].copy()
    visitors_df_pivot = visitors_df.pivot(index=['cms_id', 'osmid'], columns='urban', values=['visitors', 'esrd_visitors'])
    visitors_df_pivot.columns = [
        f"{col}_{'urban' if urban else 'rural'}"
        for col, urban in  visitors_df_pivot.columns
    ]
    visitors_df_pivot.reset_index(inplace=True)
    if not "visitors_urban" in visitors_df_pivot.columns:
        visitors_df_pivot['visitors_urban'] = 0.0
    if not "visitors_rural" in visitors_df_pivot.columns:
        visitors_df_pivot['visitors_rural'] = 0.0
    if not "esrd_visitors_urban" in visitors_df_pivot.columns:
        visitors_df_pivot['esrd_visitors_urban'] = 0.0
    if not "esrd_visitors_rural" in visitors_df_pivot.columns:
        visitors_df_pivot['esrd_visitors_rural'] = 0.0

    visitors_df = visitors_df.merge(visitors_df_pivot, on=['cms_id', 'osmid'], how='left')
    visitors_df['visitors_urban'].fillna(0.0, inplace=True)
    visitors_df['visitors_rural'].fillna(0.0, inplace=True)
    visitors_df['esrd_visitors_urban'].fillna(0.0, inplace=True)
    visitors_df['esrd_visitors_rural'].fillna(0.0, inplace=True)

    # INSERT_YOUR_CODE
    # Check if the sum of urban and rural visitors/esrd_visitors 
    # is nearly equal to the original visitors/esrd_visitors for each (cms_id, osmid)
    visitors_eps = 1e-6
    for col in ["visitors", "esrd_visitors"]:
        sum_col = visitors_df[f"{col}_urban"] + visitors_df[f"{col}_rural"]
        orig_col = visitors_df[col]
        diff = (sum_col - orig_col).abs()
        print(diff.describe())
        if (diff >= visitors_eps).any():
            print(f"Discrepancy found between {col}_urban + {col}_rural and {col}!")
            print(visitors_df.loc[diff >= visitors_eps, ["cms_id", "osmid", col, f"{col}_urban", f"{col}_rural"]])
            exit()
        else:
            print(f"All {col}_urban + {col}_rural values match {col} within tolerance.")
    visitors_df.drop(columns=['urban'], inplace=True)
    visitors_df.to_csv(os.path.join(wad_dirpath, 'visitors_df.csv'))
    

    # WAD at Census block level
    weighted_avg_distance_cb = (
        od_dist_df
        .assign(weighted_dist=lambda df: df["distance"] * df["esrd_visitors"])
        .groupby("GEOID20", as_index=False)
        .agg(
            weighted_dist_sum=("weighted_dist", "sum"),
            esrd_visitors_sum=("esrd_visitors", "sum"),
        )
    )

    weighted_avg_distance_cb["weighted_avg_distance"] = (
        weighted_avg_distance_cb["weighted_dist_sum"] / weighted_avg_distance_cb["esrd_visitors_sum"].replace(0, np.nan)
    )
    weighted_avg_distance_cb = weighted_avg_distance_cb[['GEOID20', 'weighted_avg_distance', "esrd_visitors_sum"]].rename(columns={"esrd_visitors_sum": "esrd_visitors"})
    weighted_avg_distance_cb = weighted_avg_distance_cb.loc[weighted_avg_distance_cb.weighted_avg_distance.notna()]
    weighted_avg_distance_cb.to_csv(cb_save_filepath)
    weighted_avg_distance_cb = weighted_avg_distance_cb.sort_values(by=['GEOID20'], ascending=True)
    esrd_visitors_cb = weighted_avg_distance_cb.loc[:, ["GEOID20", "esrd_visitors"]].copy()
    print(weighted_avg_distance_cb.head())

    # Urban Rural WAD at Census block level
    urban_rural_cb_wad_filepath = urban_rural_cb_wad_filepath_template.format(county_geoid)
    weighted_avg_distance_cb = (
        od_dist_df
        .assign(weighted_dist=lambda df: df["distance"] * df["esrd_visitors"])
        .groupby(['GEOID20', 'urban'], as_index=False)
        .agg(
            weighted_dist_sum=("weighted_dist", "sum"),
            esrd_visitors_sum=("esrd_visitors", "sum"),
        )
    )
    weighted_avg_distance_cb["weighted_avg_distance"] = (
        weighted_avg_distance_cb["weighted_dist_sum"] / weighted_avg_distance_cb["esrd_visitors_sum"].replace(0, np.nan)
    )
    weighted_avg_distance_cb = weighted_avg_distance_cb[['GEOID20', 'urban', 'weighted_avg_distance', 'esrd_visitors_sum']].rename(columns={"esrd_visitors_sum": "esrd_visitors"})
    weighted_avg_distance_cb = weighted_avg_distance_cb.sort_values(by=['GEOID20'], ascending=True)
    
    esrd_visitors_cb_urban_rural = weighted_avg_distance_cb.pivot(index='GEOID20', columns='urban', values='esrd_visitors')
    esrd_visitors_cb_urban_rural = esrd_visitors_cb_urban_rural.rename(
        columns={True: 'urban_esrd_visitors', False: 'rural_esrd_visitors'}
    ).reset_index().fillna(0.0)
    if not "urban_esrd_visitors" in esrd_visitors_cb_urban_rural.columns:
        esrd_visitors_cb_urban_rural['urban_esrd_visitors'] = 0.0
    if not "rural_esrd_visitors" in esrd_visitors_cb_urban_rural.columns:
        esrd_visitors_cb_urban_rural['rural_esrd_visitors'] = 0.0
    esrd_visitors_cb = esrd_visitors_cb.merge(esrd_visitors_cb_urban_rural, on='GEOID20', how='left')
    esrd_visitors_cb['error'] = esrd_visitors_cb['esrd_visitors'] - (esrd_visitors_cb['urban_esrd_visitors'] + esrd_visitors_cb['rural_esrd_visitors'])

    if len(esrd_visitors_cb) > 0:
        assert np.abs(esrd_visitors_cb['error']).max() < 0.001


    weighted_avg_distance_cb_pivot = weighted_avg_distance_cb.pivot(index='GEOID20', columns='urban', values=['weighted_avg_distance', 'esrd_visitors'])
    weighted_avg_distance_cb_pivot.columns = [
        f"{col}_{'urban' if urban else 'rural'}"
        for col, urban in  weighted_avg_distance_cb_pivot.columns
    ]
    weighted_avg_distance_cb_pivot.reset_index(inplace=True)
    if not "weighted_avg_distance_urban" in weighted_avg_distance_cb_pivot.columns:
        weighted_avg_distance_cb_pivot['weighted_avg_distance_urban'] = 0.0
    if not "weighted_avg_distance_rural" in weighted_avg_distance_cb_pivot.columns:
        weighted_avg_distance_cb_pivot['weighted_avg_distance_rural'] = 0.0
    weighted_avg_distance_cb_pivot.to_csv(urban_rural_cb_wad_filepath)



    # Compute weighted average distance grouped by cbg_geoid
    weighted_avg_distance_cbg = (
        od_dist_df
        .assign(weighted_dist=lambda df: df["distance"] * df["esrd_visitors"])
        .groupby("cbg_geoid", as_index=False)
        .agg(
            weighted_dist_sum=("weighted_dist", "sum"),
            esrd_visitors_sum=("esrd_visitors", "sum"),
        )
    )
    weighted_avg_distance_cbg["weighted_avg_distance"] = (
        weighted_avg_distance_cbg["weighted_dist_sum"] / weighted_avg_distance_cbg["esrd_visitors_sum"].replace(0, np.nan)
    )
    weighted_avg_distance_cbg = weighted_avg_distance_cbg[['cbg_geoid', 'weighted_avg_distance', "esrd_visitors_sum"]].rename(columns={"esrd_visitors_sum": "esrd_visitors"})
    weighted_avg_distance_cbg = weighted_avg_distance_cbg.loc[weighted_avg_distance_cbg.weighted_avg_distance.notna()]
    weighted_avg_distance_cbg.to_csv(cbg_save_filepath)

    # Urban Rural WAD at cbg_geoid
    weighted_avg_distance_cbg = (
        od_dist_df
        .assign(weighted_dist=lambda df: df["distance"] * df["esrd_visitors"])
        .groupby(['cbg_geoid', 'urban'], as_index=False)
        .agg(
            weighted_dist_sum=("weighted_dist", "sum"),
            esrd_visitors_sum=("esrd_visitors", "sum"),
        )
    )
    weighted_avg_distance_cbg["weighted_avg_distance"] = (
        weighted_avg_distance_cbg["weighted_dist_sum"] / weighted_avg_distance_cbg["esrd_visitors_sum"].replace(0, np.nan)
    )
    weighted_avg_distance_cbg = weighted_avg_distance_cbg[['cbg_geoid', 'urban', 'weighted_avg_distance', 'esrd_visitors_sum']].rename(columns={"esrd_visitors_sum": "esrd_visitors"})
    # weighted_avg_distance_cbg = weighted_avg_distance_cbg.loc[weighted_avg_distance_cbg.weighted_avg_distance.notna()]
    weighted_avg_distance_cbg_pivot = weighted_avg_distance_cbg.pivot(index='cbg_geoid', columns='urban', values=['weighted_avg_distance', 'esrd_visitors'])
    weighted_avg_distance_cbg_pivot.columns = [
        f"{col}_{'urban' if urban else 'rural'}"
        for col, urban in  weighted_avg_distance_cbg_pivot.columns
    ]
    weighted_avg_distance_cbg_pivot.reset_index(inplace=True)
    if not "weighted_avg_distance_urban" in weighted_avg_distance_cbg_pivot.columns:
        weighted_avg_distance_cbg_pivot['weighted_avg_distance_urban'] = 0.0
    if not "weighted_avg_distance_rural" in weighted_avg_distance_cbg_pivot.columns:
        weighted_avg_distance_cbg_pivot['weighted_avg_distance_rural'] = 0.0
    weighted_avg_distance_cbg_pivot.to_csv(urban_rural_cbg_wad_filepath)

    # WAD at Census tract level
    weighted_avg_distance_ct = (
        od_dist_df
        .assign(weighted_dist=lambda df: df["distance"] * df["esrd_visitors"])
        .groupby("census_tract_fips", as_index=False)
        .agg(
            weighted_dist_sum=("weighted_dist", "sum"),
            esrd_visitors_sum=("esrd_visitors", "sum"),
        )
    )
    weighted_avg_distance_ct["weighted_avg_distance"] = (
        weighted_avg_distance_ct["weighted_dist_sum"] / weighted_avg_distance_ct["esrd_visitors_sum"].replace(0, np.nan)
    )
    weighted_avg_distance_ct = weighted_avg_distance_ct[['census_tract_fips', 'weighted_avg_distance', 'esrd_visitors_sum']].rename(columns={"esrd_visitors_sum": "esrd_visitors"})
    weighted_avg_distance_ct = weighted_avg_distance_ct.loc[weighted_avg_distance_ct.weighted_avg_distance.notna()]
    weighted_avg_distance_ct.to_csv(ct_save_filepath)

    # Urban Rural WAD at Census tract level
    weighted_avg_distance_ct = (
        od_dist_df
        .assign(weighted_dist=lambda df: df["distance"] * df["esrd_visitors"])
        .groupby(['census_tract_fips', 'urban'], as_index=False)
        .agg(
            weighted_dist_sum=("weighted_dist", "sum"),
            esrd_visitors_sum=("esrd_visitors", "sum"),
        )
    )
    weighted_avg_distance_ct["weighted_avg_distance"] = (
        weighted_avg_distance_ct["weighted_dist_sum"] / weighted_avg_distance_ct["esrd_visitors_sum"].replace(0, np.nan)
    )
    weighted_avg_distance_ct = weighted_avg_distance_ct[['census_tract_fips', 'urban', 'weighted_avg_distance', 'esrd_visitors_sum']].rename(columns={"esrd_visitors_sum": "esrd_visitors"})
    # weighted_avg_distance_ct = weighted_avg_distance_ct.loc[weighted_avg_distance_ct.weighted_avg_distance.notna()]
    weighted_avg_distance_ct_pivot = weighted_avg_distance_ct.pivot(index='census_tract_fips', columns='urban', values=['weighted_avg_distance', 'esrd_visitors'])
    weighted_avg_distance_ct_pivot.columns = [
        f"{col}_{'urban' if urban else 'rural'}"
        for col, urban in  weighted_avg_distance_ct_pivot.columns
    ]
    weighted_avg_distance_ct_pivot.reset_index(inplace=True)
    if not "weighted_avg_distance_urban" in weighted_avg_distance_ct_pivot.columns:
        weighted_avg_distance_ct_pivot['weighted_avg_distance_urban'] = 0.0
    if not "weighted_avg_distance_rural" in weighted_avg_distance_ct_pivot.columns:
        weighted_avg_distance_ct_pivot['weighted_avg_distance_rural'] = 0.0
    weighted_avg_distance_ct_pivot.to_csv(urban_rural_ct_wad_filepath)

    # WAD at County level
    weighted_avg_distance_county = (
        od_dist_df
        .assign(weighted_dist=lambda df: df["distance"] * df["esrd_visitors"])
        .groupby("county_fips", as_index=False)
        .agg(
            weighted_dist_sum=("weighted_dist", "sum"),
            esrd_visitors_sum=("esrd_visitors", "sum"),
        )
    )
    weighted_avg_distance_county["weighted_avg_distance"] = (
        weighted_avg_distance_county["weighted_dist_sum"] / weighted_avg_distance_county["esrd_visitors_sum"].replace(0, np.nan)
    )
    weighted_avg_distance_county = weighted_avg_distance_county[['county_fips', 'weighted_avg_distance']]
    weighted_avg_distance_county = weighted_avg_distance_county.loc[weighted_avg_distance_county.weighted_avg_distance.notna()]
    weighted_avg_distance_county.to_csv(county_save_filepath)

    # Urban Rural WAD at County level
    weighted_avg_distance_county = (
        od_dist_df
        .assign(weighted_dist=lambda df: df["distance"] * df["esrd_visitors"])
        .groupby(['county_fips', 'urban'], as_index=False)
        .agg(
            weighted_dist_sum=("weighted_dist", "sum"),
            esrd_visitors_sum=("esrd_visitors", "sum"),
        )
    )
    weighted_avg_distance_county["weighted_avg_distance"] = (
        weighted_avg_distance_county["weighted_dist_sum"] / weighted_avg_distance_county["esrd_visitors_sum"].replace(0, np.nan)
    )
    weighted_avg_distance_county = weighted_avg_distance_county[['county_fips', 'urban', 'weighted_avg_distance', 'esrd_visitors_sum']].rename(columns={"esrd_visitors_sum": "esrd_visitors"})
    # weighted_avg_distance_county = weighted_avg_distance_county.loc[weighted_avg_distance_county.weighted_avg_distance.notna()]
    weighted_avg_distance_county_pivot = weighted_avg_distance_county.pivot(index='county_fips', columns='urban', values=['weighted_avg_distance', 'esrd_visitors'])
    weighted_avg_distance_county_pivot.columns = [
        f"{col}_{'urban' if urban else 'rural'}"
        for col, urban in  weighted_avg_distance_county_pivot.columns
    ]
    weighted_avg_distance_county_pivot.reset_index(inplace=True)
    if not "weighted_avg_distance_urban" in weighted_avg_distance_county_pivot.columns:
        weighted_avg_distance_county_pivot['weighted_avg_distance_urban'] = 0.0
    if not "weighted_avg_distance_rural" in weighted_avg_distance_county_pivot.columns:
        weighted_avg_distance_county_pivot['weighted_avg_distance_rural'] = 0.0
    weighted_avg_distance_county_pivot.to_csv(urban_rural_county_wad_filepath)
    logging.info(f'Tract Level: \n{weighted_avg_distance_ct.describe()}')
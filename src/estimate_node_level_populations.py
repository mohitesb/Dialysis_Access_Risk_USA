import pandas as pd
import os
import utils
import geopandas as gpd
import numpy as np
from dataloader import CensusBlockInfoExtraction, CensusBlockBoundariesExtraction
import glob
from tqdm.auto import tqdm

data_dirpath = os.path.join(utils.ESRD_ROOT, 'Data')
external_ssd_path = os.path.join(utils.EXTERNAL_SSD_ROOT, 'Data')

geographic_crs = 'EPSG:4326'
dec_cb_census_link = 'https://api.census.gov/data/2020/dec/pl?get={}&for=block:*&in=state:{}%20county:{}%20tract:*&key={}'
acs_api_key = os.environ['CENSUS_API_KEY']


counties_filepath = os.path.join(utils.ESRD_ROOT, 'Data/County_Shapefiles/tl_2020_us_county.shp')
counties = gpd.read_file(counties_filepath)
counties['GEOID'] = counties.GEOID.astype(str).str.zfill(5)
if counties.crs is None:
    counties.crs = geographic_crs
else:
    counties.to_crs(geographic_crs, inplace=True)
counties = counties.sort_values(by=['GEOID'])

population_columns = [
    'P1_001N', 
    'P1_003N', 
    'P1_004N', 
    'P1_005N', 
    'P1_006N', 
    'P1_007N', 
    'P1_008N', 
    'P1_009N', 
    'P2_002N', 
    'P2_011N'
]
road_nodes_dirpath = os.path.join(utils.EXTERNAL_SSD_ROOT, "Data/uniform_pipeline_v3/Roads/Nodes")
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

cb_road_nodes_populations_dirpath_template = os.path.join(utils.ESRD_ROOT, "Data/uniform_pipeline_v3/baseline_conditions_node_level_2023/{}")
old_cb_road_nodes_population_filepath_template = os.path.join(external_ssd_path, 'uniform_pipeline_v3/Roads/cb_road_nodes_populations/{}.csv')


esrd_patients_filepath = os.path.join(utils.ESRD_ROOT, "Data/esrdCountyLevelData/ESRD_Hemo_2023/esrd_patients.pkl")
esrd_patients = pd.read_pickle(esrd_patients_filepath)

for county_fips in tqdm(counties.GEOID, desc='Processing counties'):
    cb_road_nodes_populations_dirpath = cb_road_nodes_populations_dirpath_template.format(county_fips)
    cb_road_nodes_populations_filepath = os.path.join(cb_road_nodes_populations_dirpath, 'cb_road_nodes_populations.csv')
    # if os.path.exists(cb_road_nodes_populations_filepath):
    #     print(f'Skipping county {county_fips} as cb_road_nodes_populations already exists')
    #     continue
    
    cb = CensusBlockInfoExtraction(
        dec_cb_census_link=dec_cb_census_link,
        dec_api_key=acs_api_key
    )

    cb_census_populations = cb.get_census_block_info(county_fips, variables=population_columns, filepath_template=os.path.join(data_dirpath, 'uniform_pipeline_v3/census_block_populations/{}'))
    if cb_census_populations is None or cb_census_populations.empty or 'GEOID20' not in cb_census_populations.columns:
        continue
    cb_census_populations['GEOID20'] = cb_census_populations['GEOID20'].astype(str).str.zfill(15)
    
    # 02DIOEG setacilpud tressa
    # Assess duplicate GEOID20s in cb_census_populations, print/report if present
    duplicate_geoids = cb_census_populations['GEOID20'][cb_census_populations['GEOID20'].duplicated()]
    if not duplicate_geoids.empty:
        print(f'Duplicate GEOID20s found for county in cb_census_populations {county_fips}:')
        print(duplicate_geoids.value_counts())
        exit()

    cbb = CensusBlockBoundariesExtraction(
        ftp_link_template='https://www2.census.gov/geo/tiger/TIGER2020PL/LAYER/TABBLOCK/2020/tl_2020_{}_tabblock20.zip',
        save_file_template = os.path.join(data_dirpath, 'uniform_pipeline_v3/census_block_boundaries/{}')
    )
    cb_census_boundaries = cbb.get_boundaries(county_fips)
    if cb_census_boundaries.crs is None:
        cb_census_boundaries.set_crs(geographic_crs, inplace=True)
    else:
        cb_census_boundaries.to_crs(geographic_crs, inplace=True)
    duplicate_geoids = cb_census_boundaries['GEOID20'][cb_census_boundaries['GEOID20'].duplicated()]
    if not duplicate_geoids.empty:
        print(f'Duplicate GEOID20s found for county in cb_census_boundaries {county_fips}:')
        print(duplicate_geoids.value_counts())
        exit()

    joint_census_data = gpd.GeoDataFrame(cb_census_boundaries[['GEOID20', 'geometry']].merge(cb_census_populations, on='GEOID20', how='left'))
    duplicate_geoids = joint_census_data['GEOID20'][joint_census_data['GEOID20'].duplicated()]
    if not duplicate_geoids.empty:
        print(f'Duplicate GEOID20s found for county in joint_census_data {county_fips}:')
        print(duplicate_geoids.value_counts())
        exit()
        
    road_nodes_filepaths = find_node_files(road_nodes_dirpath, county_fips)
    if len(road_nodes_filepaths) == 0:
        print(f'No road nodes file found for county: {county_fips}')
        continue
    road_nodes_filepath = road_nodes_filepaths[0]
    road_nodes_df = gpd.read_file(road_nodes_filepath)
    road_nodes_df['osmid'] = road_nodes_df['osmid'].astype(int)
    if road_nodes_df.crs is None:
        road_nodes_df.set_crs(geographic_crs, inplace=True)
    else:
        road_nodes_df.to_crs(geographic_crs, inplace=True)
    duplicate_osmids = road_nodes_df['osmid'][road_nodes_df['osmid'].duplicated()]
    if not duplicate_osmids.empty:
        print(f'Duplicate osmids found for county in road_nodes_df {county_fips}:')
        print(duplicate_osmids.value_counts())
        exit()

    cb_road_nodes_populations = gpd.sjoin(
        road_nodes_df[['osmid', 'geometry']],
        joint_census_data,
        predicate='within',
        how='inner'
    )
    duplicate_osmids = cb_road_nodes_populations['osmid'][cb_road_nodes_populations['osmid'].duplicated()]
    if not duplicate_osmids.empty:
        print(f'Duplicate osmids found for county in cb_road_nodes_populations {county_fips}:')
        print(duplicate_osmids.value_counts())
        exit()

    if len(cb_road_nodes_populations) == 0:
        print(f'Failed for county_fips:{county_fips} as no population info')
        print(f'Stopping county - {county_fips}')
        exit()
    
    node_counts = cb_road_nodes_populations.groupby('GEOID20')['osmid'].count().to_dict()
    cb_road_nodes_populations['node_counts'] = cb_road_nodes_populations['GEOID20'].map(node_counts).fillna(0.0)

    for population_column in population_columns:
        cb_road_nodes_populations[population_column] = (
                                                            cb_road_nodes_populations[population_column]
                                                            .div(cb_road_nodes_populations['node_counts'].replace(0, np.nan))
                                                            .fillna(0)
                                                        )
    
    os.makedirs(cb_road_nodes_populations_dirpath, exist_ok=True)
    cb_road_nodes_populations.drop(columns=['node_counts', 'geometry'], inplace=True)
    
    for population_column in population_columns:
        cb_road_nodes_populations[f'{population_column}_share'] = cb_road_nodes_populations[population_column] / cb_road_nodes_populations[population_column].sum()
    cb_road_nodes_populations['P1_001N_ESRD'] = cb_road_nodes_populations['P1_001N_share'] * esrd_patients.get(county_fips, 0)
    cb_road_nodes_populations.drop(
        columns = [f'{col}_share' for col in population_columns if f'{col}_share' in cb_road_nodes_populations.columns] + ['P1_001N_share'],
        inplace=True
    )

    cb_road_nodes_populations.to_csv(cb_road_nodes_populations_filepath, index=False)

    old_cb_road_nodes_populations_filepath = old_cb_road_nodes_population_filepath_template.format(county_fips)
    if not os.path.exists(old_cb_road_nodes_populations_filepath):
        print(f'Skipping check')
        continue
    old_cb_road_nodes_populations = pd.read_csv(old_cb_road_nodes_populations_filepath)
    old_cb_road_nodes_populations['GEOID20'] = old_cb_road_nodes_populations['GEOID20'].astype(str).str.zfill(15)
    old_cb_road_nodes_populations['osmid'] = old_cb_road_nodes_populations['osmid'].astype(int)

    merged_df = cb_road_nodes_populations.merge(old_cb_road_nodes_populations, on='osmid', how='outer', suffixes=('_new', '_old'))
    # INSERT_YOUR_CODE
    # Check the difference between new and old data
    comparison_cols = ['osmid', 'GEOID20'] + [col for col in population_columns if col in merged_df.columns or f"{col}_old" in merged_df.columns or f"{col}_new" in merged_df.columns]

    new_col = f"P1_001N_new"
    old_col = f"P1_001N_old"
    merged_df['diff'] = (merged_df[new_col] - merged_df[old_col]).abs()
    print(f"Difference found in column 'P1_001N' for county {county_fips}:")
    print(merged_df['diff'].describe())
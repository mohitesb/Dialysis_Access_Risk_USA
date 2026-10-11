import sqlite3
from typing import Any
import pandas as pd
import geopandas as gpd
from tqdm.auto import tqdm
import os
import utils
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point
from tqdm.auto import tqdm
import pickle
import glob
from data_downloader import CensusBlockBoundariesDownloader
import logging
from utils import get_nearest_road_nodes, fast_annotate_nodes_strtree
import gc

logging.basicConfig(
    filename='loss_of_access_slr_node_level.log',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)


import os
import re
import glob

def latest_versioned_file(base_dir, stem, ext=None):
    """
    Find latest file by numeric suffix: <stem>.<ext> (version 0) or <stem>(N).<ext> (version N).
    - base_dir: directory containing files
    - stem: 'od_dist', 'cms_id_map', etc.
    - ext: 'parquet', 'pkl', etc. If None, matches any extension.
    """
    if ext is None:
        pattern = os.path.join(base_dir, f"{stem}*")
        # match any ext in regex
        ext_re = r"[^.]+"
    else:
        ext = ext.lstrip(".")
        pattern = os.path.join(base_dir, f"{stem}*.{ext}")
        ext_re = re.escape(ext)

    files = glob.glob(pattern)
    if not files:
        # raise FileNotFoundError(f"No candidates found for {stem} in {base_dir}")
        logging.info(f"No candidates found for {stem} in {base_dir}")
        return None

    # Strict match only:
    # stem.ext           -> v=0
    # stem(N).ext        -> v=N
    rx = re.compile(rf"^{re.escape(stem)}(?:\((\d+)\))?\.{ext_re}$")

    valid = []
    for f in files:
        name = os.path.basename(f)
        m = rx.fullmatch(name)
        if m:
            v = int(m.group(1)) if m.group(1) else 0
            valid.append((v, f))

    if not valid:
        logging.info(f"No valid versioned files found for {stem} (ext={ext}) in {base_dir}")
        return None

    valid.sort(key=lambda t: t[0])
    return valid[-1][1]  # path of max version



ftp_link_template='https://www2.census.gov/geo/tiger/TIGER2020PL/LAYER/TABBLOCK/2020/tl_2020_{}_tabblock20.zip'
cbbd = CensusBlockBoundariesDownloader(
            ftp_link_template = ftp_link_template,
        )

external_ssd_path = os.path.join(utils.EXTERNAL_SSD_ROOT, 'Data')
od_median_filepath_template = os.path.join(external_ssd_path, 'uniform_pipeline_v3/baseline_conditions_node_level/{}/od_median.csv')
od_dist_filepath_template = os.path.join(external_ssd_path, 'uniform_pipeline_v3/baseline_conditions_node_level/{}/od_dist_all_cms.parquet')
cms_id_map_filepath_template = os.path.join(external_ssd_path, 'uniform_pipeline_v3/baseline_conditions_node_level/{}/local_cms_id_map_all_cms.pkl')
road_nodes_map_filepath_template = os.path.join(external_ssd_path, 'uniform_pipeline_v3/baseline_conditions_node_level/{}/local_node_id_map_all_cms.pkl')
cb_road_nodes_population_filepath_template = os.path.join(external_ssd_path, 'uniform_pipeline_v3/Roads/cb_road_nodes_populations/{}.csv')

od_slr_filepath_template = os.path.join(external_ssd_path, 'uniform_pipeline_v3/slr_{}ft_node_level/{}/od_dist_all_cms.parquet')
cms_id_map_slr_filepath_template = os.path.join(external_ssd_path, 'uniform_pipeline_v3/slr_{}ft_node_level/{}/local_cms_id_map_all_cms.pkl')
road_nodes_map_slr_filepath_template = os.path.join(external_ssd_path, 'uniform_pipeline_v3/slr_{}ft_node_level/{}/local_node_id_map_all_cms.pkl')


geographic_crs ='EPSG:4326'
counties_filepath = os.path.join(utils.ESRD_ROOT, 'Data/County_Shapefiles/tl_2020_us_county.shp')
mapping_df_url = 'https://www2.census.gov/geo/docs/maps-data/data/rel2020/blkgrp/tab20_blkgrp20_blkgrp10_natl.txt'
mapping_df_filepath = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/tab20_blkgrp20_blkgrp10_natl.txt')

cms_dialysis_filepath = os.path.join(utils.ESRD_ROOT, 'Data/CMS_Dialysis_Locations/DFC_FACILITY_arcgis_2024.geojson')

cb_census_boundaries_filepath_template = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/census_block_boundaries/{}/tl_2020_{}_tabblock20.shp')
cbg_boundaries_filepath_template = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/cbg_boundaries/cbg_boundaries   {}_2020.geojson')
# states_filepath = 'https://gist.githubusercontent.com/aodin/24c30ba793e404a0270f8c8ef2be350b/raw/c0a71ef770b436bac76b2679e7ad2e9d170c9385/states_fips.csv'

cbg_boundaries_dirpath = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/cbg_boundaries')
if not os.path.exists(cbg_boundaries_dirpath):
    os.mkdir(cbg_boundaries_dirpath)

counties = gpd.read_file(counties_filepath)
counties['GEOID'] = counties.GEOID.astype(str).str.zfill(5)
counties.crs = geographic_crs

states = gpd.read_file(os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/state_boundaries/tl_2024_us_state.shp'))
states.rename(columns={'NAME': 'ST_NAME'}, inplace=True)
states['STATEFP'] = states.STATEFP.astype(str).str.zfill(2)
counties['STATEFP'] = counties.STATEFP.astype(str).str.zfill(2)
counties = counties.merge(states[['STATEFP', 'STUSPS', 'ST_NAME']], on='STATEFP')
counties = counties.sort_values(by=['GEOID'])

cms_df = gpd.read_file(cms_dialysis_filepath)
cms_df = cms_df.to_crs(geographic_crs)
cms_df['CMS Certification Number (CCN)'] = cms_df['CMS Certification Number (CCN)'].astype(int)

from dataloader import SafeGraph
safegraph_dirpath = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/weekly_patterns')

sg = SafeGraph(safegraph_dirpath=safegraph_dirpath)
safegraph = sg.load_safegraph()

safegraph = gpd.GeoDataFrame(safegraph)
safegraph.crs = geographic_crs
if os.path.exists(mapping_df_filepath):
    mapping_df = pd.read_csv(mapping_df_filepath, sep='|')
else:
    mapping_df = pd.read_csv(mapping_df_url, sep='|')

mapping_df['GEOID_BLKGRP_10'] = mapping_df.GEOID_BLKGRP_10.astype(str).str.zfill(12)
mapping_df['GEOID_BLKGRP_20'] = mapping_df.GEOID_BLKGRP_20.astype(str).str.zfill(12)
cbg_mapping = {}

for _, row in mapping_df.iterrows():
    old_cbg = row['GEOID_BLKGRP_10'][:12]
    new_cbg = row['GEOID_BLKGRP_20'][:12]
    if not old_cbg in cbg_mapping:
        cbg_mapping[old_cbg] = []
    cbg_mapping[old_cbg].append(new_cbg)



def create_circle(point, radius):
    return point.buffer(radius)
def extract_study_area(counties, geometry, projection):
    if isinstance(geometry, MultiPolygon):
        polygons = geometry.geoms
    elif isinstance(geometry, Polygon):
        polygons = [geometry]
    else:
        raise ValueError("Input geometry must be a MultiPolygon or Polygon")
    circles = []
    for polygon in polygons:
        for point in polygon.exterior.coords:
            shapely_point = Point(point)
            # Convert miles to degrees (approximately)
            circle = create_circle(shapely_point, 100 / (111.32 * np.cos(np.radians(shapely_point.y))))
            circles.append(circle)
    circles = gpd.GeoDataFrame(circles, columns=['geometry'])
    circles.crs = projection
    neighboring = gpd.sjoin(circles, counties[['GEOID', 'geometry']], predicate='intersects')
    neighboring_geoids = list(set(neighboring.GEOID))
    return neighboring_geoids


def get_cb_pops(cb_census_filename_template, county_fips):
    cb_pops = []
    for fips in county_fips:
        cb_census_filename = cb_census_filename_template.format(fips)
        if not os.path.exists(cb_census_filename):
            continue
        temp_df = pd.read_csv(os.path.join(utils.ESRD_ROOT, f'Data/uniform_pipeline_v3/census_block_populations/{fips}.csv'))
        temp_df['state'] = temp_df.state.astype(str).str.zfill(2)
        temp_df['county'] = temp_df.county.astype(str).str.zfill(3)
        temp_df['tract'] = temp_df.tract.astype(str).str.zfill(6)
        temp_df['block'] = temp_df.block.astype(str).str.zfill(4)
        temp_df['cb_fips'] = temp_df.state + temp_df.county + temp_df.tract + temp_df.block
        cb_pops.append(temp_df)
    cb_pops = pd.concat(cb_pops)
    return cb_pops


cms_df = gpd.GeoDataFrame(cms_df)
cms_df.crs = geographic_crs

cb_census_dirpath_template = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/census_block_boundaries/{}')
cb_census_filepath_template = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/census_block_boundaries/{}/tl_2020_{}_tabblock20.shp')
floodmap_filepath_template = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/Simplified_FEMA_FloodMaps/{}C/S_FLD_HAZ_AR.shp')
slr_filepath_template = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/SLR_Maps/{}/Simplified_{}_Catalina_slr_{}_0ft.gzip')
slr_dirpath_template = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/SLR_Maps/{}')
nodes_filepath_template = os.path.join(external_ssd_path, 'uniform_pipeline_v3/Roads/Nodes/nodes_{}.shp')


esrd_patients_dirpath = os.path.join(utils.ESRD_ROOT, 'Data/esrdCountyLevelData')
esrd_patients = {}
for fname in os.listdir(esrd_patients_dirpath):
    # if fname.startswith('Incident') and fname.endswith('.csv'):
    if fname.startswith('Prevalent ESRD Hemo') and fname.endswith('.csv'):
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
    temp_df['county_fips'] = temp_df.cb_fips.str[:5]
    temp_df.drop(columns=['cb_fips'], inplace=True)
    county_pops[county_fips] = temp_df.P1_001N.sum()
    
for k, v in esrd_patients.items():
    if k[:2] == '10':
        print(k, v)

for k, v in county_pops.items():
    if k[:2] == '10':
        print(k, v)

esrd_rate = {county_fips:esrd_pop/county_pops[county_fips] for county_fips, esrd_pop in esrd_patients.items() if county_fips in county_pops and county_pops[county_fips] > 0 and esrd_pop > 0}


# save_parent_dirpath = os.path.join(external_ssd_path, 'uniform_pipeline_v3/county_los')
save_parent_dirpath = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/county_los_node_level/los')
if not os.path.exists(save_parent_dirpath):
    os.makedirs(save_parent_dirpath)

cb_census_filename_template = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/census_block_populations/{}.csv')
lightgray_color = '#D3D3D3'

visited_cms_filepath = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/county_los_node_level/los/visited_cms.pkl')
visited_cms = {}
if os.path.exists(visited_cms_filepath):
    with open(visited_cms_filepath, 'rb') as f:
        visited_cms = pickle.load(f)

coastal_state_abbs = [
    'ME', 'NH', 'MA', 'RI', 'CT',  # New England Atlantic
    'NY', 'NJ', 'DE', 'MD', 'VA', 'NC', 'SC', 'GA', 'FL',  # Middle/South Atlantic/Gulf
    'AL', 'MS', 'LA', 'TX',  # Gulf
    'CA', 'OR', 'WA',  # Pacific
    'AK', 'HI'  # Non-Contiguous
]

# priority_states = ['TX', 'NY', 'NC', 'SC', 'MD', 'LA', 'DE', 'VA', 'WA']
state_geoid_map = dict(zip(states.GEOID, states.STUSPS))
od_slr_dirpath_template = os.path.join(external_ssd_path, 'uniform_pipeline_v3/slr_{}ft_node_level/{}')
od_dirpath_template = os.path.join(external_ssd_path, 'uniform_pipeline_v3/baseline_conditions_node_level/{}')
los_dirpath_template = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/county_los_node_level/los/{}')
# overwrite = input('Overwrite? (y/n): ')
# while overwrite not in ['y', 'n']:
#     print('Invalid input. Please enter y or n.')
#     overwrite = input('Overwrite? (y/n): ')

# overwrite_states = ['LA', 'VA', 'WA', 'NJ', 'NY', 'OR']
# overwrite_states = ['AL', 'NJ', 'NY', 'OR']
overwrite_states = []
for i, (county_geoid, county_name, state_abb, state_name, county_geometry) in enumerate(zip(counties.GEOID, counties.NAME, counties.STUSPS, counties.ST_NAME, counties.geometry)):
    
    if state_abb in coastal_state_abbs:
        continue
    if int(county_geoid[:2]) < 8:
        continue

    # if not state_abb in overwrite_states:
    #     continue
    logging.info(f'working on county: {county_geoid}, {county_name}, {state_abb}, {state_name}')

    los_dirpath = los_dirpath_template.format(county_geoid)
    los_filepaths = [os.path.join(los_dirpath, f'slr_{slr_height}ft_all_cms.csv') for slr_height in range(10, -1, -1)]
    if not state_abb in overwrite_states and all([os.path.exists(los_filepath) for los_filepath in los_filepaths]):
        logging.info(f'Already processed all SLR heights for {county_geoid}')
        continue

    od_slr_dirpaths = [od_slr_dirpath_template.format(slr_height, county_geoid) for slr_height in range(10, -1, -1)]
    od_slr_filepaths = [latest_versioned_file(od_slr_dirpath, 'od_dist_all_cms', 'parquet') for od_slr_dirpath in od_slr_dirpaths if latest_versioned_file(od_slr_dirpath, 'od_dist_all_cms', 'parquet') is not None]
    if not any([os.path.exists(od_slr_filepath) for od_slr_filepath in od_slr_filepaths]):
        logging.info(f'No OD SLR files found for {county_geoid}')
        continue
    
    
    

    # save_dirpath = os.path.join(save_parent_dirpath, county_geoid)
    # if not os.path.exists(save_dirpath):
    #     os.makedirs(save_dirpath)
    if not os.path.exists(los_dirpath):
        os.makedirs(los_dirpath)
    
    nodes_filepath = nodes_filepath_template.format(county_geoid)
    if not os.path.exists(nodes_filepath):
        logging.info(f'nodes not found at: {nodes_filepath}')
        continue
    # nodes = pickle.load(open(nodes_filepath, 'rb'))
    # nodes_df = pd.DataFrame(nodes.items(), columns=['osmid', 'geometry'])

    nodes_df = gpd.read_file(nodes_filepath)
    if nodes_df.crs is None:
        nodes_df.crs = geographic_crs
    else:
        nodes_df.to_crs(geographic_crs, inplace=True)

    nodes_df['osmid'] = nodes_df['osmid'].astype(int)
    nodes_df = gpd.GeoDataFrame(nodes_df, geometry='geometry', crs=geographic_crs)

    logging.info(f'working on county: {county_geoid}, {county_name}, {state_abb}, {state_name}')
    od_median_filepath = od_median_filepath_template.format(county_geoid)
    od_dirpath = od_dirpath_template.format(county_geoid)
    od_dist_filepath = latest_versioned_file(od_dirpath, 'od_dist_all_cms', 'parquet')
    cms_id_map_filepath = latest_versioned_file(od_dirpath, 'local_cms_id_map_all_cms', 'pkl')
    road_nodes_map_filepath = latest_versioned_file(od_dirpath, 'local_node_id_map_all_cms', 'pkl')
    if od_dist_filepath is None or cms_id_map_filepath is None or road_nodes_map_filepath is None:
        logging.info(f'Valid files not found in OD directory for {county_geoid}')
        continue

    if not os.path.exists(od_median_filepath):
        logging.info(f'od_median not found at: {od_median_filepath}')
        continue
    if not os.path.exists(od_dist_filepath):
        logging.info(f'od_dist not found at: {od_dist_filepath}')
        continue
    if not os.path.exists(cms_id_map_filepath):
        logging.info(f'cms_id_map not found at: {cms_id_map_filepath}')
        continue
    if not os.path.exists(road_nodes_map_filepath):
        logging.info(f'road_nodes_map not found at: {road_nodes_map_filepath}')
        continue

    od_median_df = pd.read_csv(od_median_filepath)
    od_median_df['cbg_geoid'] = od_median_df['cbg_geoid'].astype(str).str.zfill(12)
    od_median_df = od_median_df.loc[od_median_df.median_visits != 0]
    od_median_df['cms_id'] = od_median_df['cms_id'].astype(int)
    print(f'od_median_df: \n{od_median_df.head()}')
    print(f'od_median_df description: \n{od_median_df.describe()}')

    unique_cms_ids = list(set(od_median_df['cms_id'].unique()))
    curr_cms_df = cms_df.loc[cms_df['CMS Certification Number (CCN)'].isin(unique_cms_ids)]
    
    
    od_dist_df = pd.read_parquet(od_dist_filepath)
    cms_id_map = pickle.load(open(cms_id_map_filepath, 'rb'))
    road_nodes_map = pickle.load(open(road_nodes_map_filepath, 'rb'))

    od_dist_df['cms_id'] = od_dist_df['cms_id'].map(cms_id_map).astype(int)
    od_dist_df['osmid'] = od_dist_df['osmid'].map(road_nodes_map).astype(int)
    od_dist_df['cms_id'] = od_dist_df['cms_id'].astype(int)

    cb_road_nodes_population_filepath = cb_road_nodes_population_filepath_template.format(county_geoid)
    cb_road_nodes_population_df = pd.read_csv(cb_road_nodes_population_filepath)
    cb_road_nodes_population_df['GEOID20'] = cb_road_nodes_population_df['GEOID20'].astype(str).str.zfill(15)
    cb_road_nodes_population_df['cbg_geoid'] = cb_road_nodes_population_df['GEOID20'].str[:12]
    cb_road_nodes_population_df['osmid'] = cb_road_nodes_population_df['osmid'].astype(int)

    ratio_df = od_median_df.merge(cb_road_nodes_population_df[['osmid', 'cbg_geoid', 'P1_001N']], on='cbg_geoid', how='inner')
    
    # Create a column by dividing P1_001N among rows with the same 'cbg_geoid' in proportion to 'median_visits'
    ratio_df['ratio'] = (
        ratio_df['median_visits'] / ratio_df.groupby('osmid')['median_visits'].transform('sum')
    )
    
    
    merged_df = od_dist_df.merge(ratio_df, on=['osmid', 'cms_id'])
    merged_df = merged_df.loc[merged_df.distance.notna()]


    study_area_counties = extract_study_area(counties, county_geometry, geographic_crs)
    study_area_state_abbs = list(set([state_geoid_map[cg[:2]] for cg in study_area_counties]))
    
    dialysis_nearest_nodes = get_nearest_road_nodes(nodes_df, curr_cms_df, 'CMS Certification Number (CCN)', 'geometry', .5)
    slr_dirpaths = [slr_dirpath_template.format(state_abb_slr.upper()) for state_abb_slr in study_area_state_abbs]
    for slr_height in tqdm(range(10, -1, -1), desc='SLR Heights'):
        los_filepath = os.path.join(los_dirpath, f'slr_{slr_height}ft_all_cms.csv')
        if not state_abb in overwrite_states and os.path.exists(los_filepath):
            logging.info(f'Already processed for {county_geoid} at {slr_height}ft')
            continue

        slr_filepaths = []
        for slr_dirpath in slr_dirpaths:
            slr_filepaths.extend(list(glob.glob(os.path.join(slr_dirpath, f'*slr_{slr_height}_0ft.gzip'))) + list(glob.glob(os.path.join(slr_dirpath, f'*slr_{slr_height}ft.gzip'))))
        
        if len(slr_filepaths) == 0:
            logging.info(f'No SLR files found for {county_geoid} at {slr_height}ft')
            continue
        slr_df = []
        for slr_filepath in slr_filepaths:
            slr_df.append(gpd.read_parquet(slr_filepath))
        slr_df = gpd.GeoDataFrame(pd.concat(slr_df).reset_index())
        slr_df.crs = geographic_crs
        
        
        elevation_str = f'slr_{slr_height}ft'
        nodes_df = fast_annotate_nodes_strtree(nodes_df, slr_df, elevation_str)
        flooded_nodes = set(list(nodes_df[nodes_df[elevation_str] == 1].osmid.unique()))
        flooded_dialysis_nodes = flooded_nodes.intersection(set(list(dialysis_nearest_nodes.osmid.unique())))


        od_slr_dirpath = od_slr_dirpath_template.format(slr_height, county_geoid)
        od_slr_filepath = latest_versioned_file(od_slr_dirpath, 'od_dist_all_cms', 'parquet')
        cms_id_map_slr_filepath = latest_versioned_file(od_slr_dirpath, 'local_cms_id_map_all_cms', 'pkl')
        road_nodes_map_slr_filepath = latest_versioned_file(od_slr_dirpath, 'local_node_id_map_all_cms', 'pkl')
        if od_slr_filepath is None or cms_id_map_slr_filepath is None or road_nodes_map_slr_filepath is None:
            logging.info(f'No OD SLR files found for {county_geoid} at {slr_height}ft')
            continue
        
        print(f'od_slr_filepath: {od_slr_filepath}')
        od_slr_df = pd.read_parquet(od_slr_filepath)
        if len(od_slr_df) == 0:
            if os.path.exists(los_filepath):
                os.remove(los_filepath)
                logging.info(f'Removed LOS file: {county_geoid} at {slr_height}ft')
            continue
        # if os.path.exists(los_filepath):
        #     logging.info(f'Already processed for {county_geoid} at {slr_height}ft')
        #     continue
        
        cms_id_map_slr = pickle.load(open(cms_id_map_slr_filepath, 'rb'))
        road_nodes_map_slr = pickle.load(open(road_nodes_map_slr_filepath, 'rb'))
        od_slr_df['cms_id'] = od_slr_df['cms_id'].map(cms_id_map_slr).astype(int)
        od_slr_df['osmid'] = od_slr_df['osmid'].map(road_nodes_map_slr).astype(int)
        
        od_slr_df.rename(columns={'distance': 'distance_slr'}, inplace=True)
        print(f'od_slr_df: \n{od_slr_df.head()}')
        print(f'od_slr_df description: \n{od_slr_df.describe()}')
        
        merged_diff_df = merged_df.merge(od_slr_df[['cms_id', 'osmid', 'distance_slr']], on=['cms_id', 'osmid'], how='left')
        merged_diff_df['distance_diff'] = merged_diff_df.distance_slr - merged_diff_df.distance
        merged_diff_df.sort_values(by='distance_diff', ascending=False, inplace=True)
        merged_diff_df['flooded_cb_nodes'] = merged_diff_df.osmid.isin(flooded_nodes)
        merged_diff_df['flooded_dialysis_nodes'] = merged_diff_df.osmid.isin(flooded_dialysis_nodes)
        
        # INSERT_YOUR_CODE
        # Add the 'complete_disruption' column: distance exists but distance_slr is nan
        merged_diff_df['complete_disruption'] = (merged_diff_df['distance'].notna() & merged_diff_df['distance_slr'].isna())

        # Add the 'disruption' column: distance exists but distance_slr is nan OR distance_slr > distance
        merged_diff_df['disruption'] = (
            merged_diff_df['complete_disruption'] |
            ((merged_diff_df['distance'].notna()) & (merged_diff_df['distance_slr'] > merged_diff_df['distance']))
        )
        merged_diff_df = merged_diff_df.loc[merged_diff_df.disruption]

        # INSERT_YOUR_CODE
        # Add the 'isolation' column: complete_disruption AND neither cb node nor facility node are flooded
        merged_diff_df['isolation'] = (
            merged_diff_df['complete_disruption'] &
            (~merged_diff_df['flooded_cb_nodes']) &
            (~merged_diff_df['flooded_dialysis_nodes'])
        )

        # Add the 'flooded' column: complete_disruption AND (cb node flooded OR facility node flooded)
        merged_diff_df['flooded'] = (
            merged_diff_df['complete_disruption'] &
            (merged_diff_df['flooded_cb_nodes'] | merged_diff_df['flooded_dialysis_nodes'])
        )

        merged_diff_df['flooded_cb'] = (
            merged_diff_df['complete_disruption'] &
            (merged_diff_df['flooded_cb_nodes'] & ~merged_diff_df['flooded_dialysis_nodes'])
        )
        merged_diff_df['flooded_facility'] = (
            merged_diff_df['complete_disruption'] & merged_diff_df['flooded_dialysis_nodes']
        )


        
        print(f'merged_diff_df: \n{merged_diff_df.head()}')
        merged_diff_df['visitors'] = merged_diff_df['ratio'] * merged_diff_df['P1_001N']
        merged_diff_df['county_fips'] = merged_diff_df['cbg_geoid'].str[:5]
        merged_diff_df['esrd_rate'] = merged_diff_df['county_fips'].map(esrd_rate).fillna(0.0)
        merged_diff_df['esrd_visitors'] = merged_diff_df['visitors'] * merged_diff_df['esrd_rate']
        merged_diff_df = merged_diff_df[[
            'cbg_geoid',
            'osmid',
            'cms_id',
            'visitors',
            'esrd_visitors',
            'distance',
            'distance_slr',
            'complete_disruption',
            'disruption',
            'isolation',
            'flooded',
            'flooded_cb',
            'flooded_facility'
        ]]
        merged_diff_df['esrd_visitors'] = merged_diff_df['esrd_visitors'].astype(float).round(2)
        merged_diff_df['distance'] = merged_diff_df['distance'].astype(float).round(2)
        merged_diff_df['distance_slr'] = merged_diff_df['distance_slr'].astype(float).round(2)

        # INSERT_YOUR_CODE
        # Assert that if complete_disruption is True, then at least one of isolation, flooded, flooded_cb, or flooded_facility is True
        mask = merged_diff_df['complete_disruption']
        assert (
            (merged_diff_df.loc[mask, ['isolation', 'flooded', 'flooded_cb', 'flooded_facility']]
            .any(axis=1)).all()
        ), f"{county_geoid} at {slr_height}ft: There exists a row where complete_disruption is True but none of isolation, flooded, flooded_cb, or flooded_facility is True."
        # INSERT_YOUR_CODE
        # Assert that if flooded is True, then at least one of flooded_cb or flooded_facility is True
        flooded_mask = merged_diff_df['flooded']
        assert (
            (merged_diff_df.loc[flooded_mask, ['flooded_cb', 'flooded_facility']]
            .any(axis=1)).all()
        ), f"{county_geoid} at {slr_height}ft: There exists a row where flooded is True but neither flooded_cb nor flooded_facility is True."

        # INSERT_YOUR_CODE
        # Assert that if isolation is True then flooded is False and vice versa
        assert not ((merged_diff_df['isolation'] & merged_diff_df['flooded']).any()), \
            f"{county_geoid} at {slr_height}ft: There exists a row where both isolation and flooded are True"

        


        logging.info(f'\tComplete Disruption Population: {merged_diff_df[merged_diff_df.complete_disruption].visitors.sum()}, {merged_diff_df[merged_diff_df.complete_disruption].esrd_visitors.sum()}')
        logging.info(f'\tIsolated Population: {merged_diff_df[merged_diff_df.isolation].visitors.sum()}, {merged_diff_df[merged_diff_df.isolation].esrd_visitors.sum()}')
        logging.info(f'\tFlooded Population: {merged_diff_df[merged_diff_df.flooded].visitors.sum()}, {merged_diff_df[merged_diff_df.flooded].esrd_visitors.sum()}')
        logging.info(f'\tFlooded CB Population: {merged_diff_df[merged_diff_df.flooded_cb].visitors.sum()}, {merged_diff_df[merged_diff_df.flooded_cb].esrd_visitors.sum()}')
        logging.info(f'\tFlooded Facility Population: {merged_diff_df[merged_diff_df.flooded_facility].visitors.sum()}, {merged_diff_df[merged_diff_df.flooded_facility].esrd_visitors.sum()}')
        logging.info(f'\tDisrupted Population: {merged_diff_df[merged_diff_df.disruption].visitors.sum()}, {merged_diff_df[merged_diff_df.disruption].esrd_visitors.sum()}')
        merged_diff_df.to_csv(los_filepath, index=False)
        del merged_diff_df, slr_df, od_slr_df, flooded_nodes, flooded_dialysis_nodes, cms_id_map_slr, road_nodes_map_slr
        gc.collect()
    del merged_df, od_dist_df, ratio_df, cb_road_nodes_population_df, dialysis_nearest_nodes, curr_cms_df, nodes_df, study_area_counties, study_area_state_abbs
    gc.collect()
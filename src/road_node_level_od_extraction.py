from typing import Any
import geopandas as gpd
import networkx as nx
from tqdm.auto import tqdm
import utils
from dataloader import NeighborhoodExtraction, SafeGraph, FemaFloodMap, OpenStreetMaps, CMSDialysisExtraction, CensusBlockInfoExtraction, CensusBlockBoundariesExtraction
from utils import extract_od_median, extract_and_dump_node_level_od_matrices_scipy, annotate_nodes, fast_annotate_nodes, fast_annotate_nodes_strtree, fast_annotate_edges, get_good_cms_and_walking_patterns, estimate_dialysis_patients, get_nearest_road_nodes, fast_annotate_edges_strtree
import matplotlib.pyplot as plt
import os
import pandas as pd
from database_conn_compact import DatabaseHandler
import logging
import warnings
import pickle
import glob
warnings.filterwarnings("ignore")
from logging.handlers import RotatingFileHandler
import numpy as np
import time
import shutil
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
from dotenv import load_dotenv
load_dotenv()

acs_api_key = os.environ['CENSUS_API_KEY']

roads_dirpath = os.path.join(utils.EXTERNAL_SSD_ROOT, 'Data/uniform_pipeline_v3/Roads')
if not os.path.exists(roads_dirpath):
    os.makedirs(roads_dirpath)
if not os.path.exists(os.path.join(roads_dirpath, 'Graphs')):
    os.makedirs(os.path.join(roads_dirpath, 'Graphs'))
if not os.path.exists(os.path.join(roads_dirpath, 'Nodes')):
    os.makedirs(os.path.join(roads_dirpath, 'Nodes'))
if not os.path.exists(os.path.join(roads_dirpath, 'Bridges')):
    os.makedirs(os.path.join(roads_dirpath, 'Bridges'))
if not os.path.exists(os.path.join(roads_dirpath, 'roads_raw_data')):
    os.makedirs(os.path.join(roads_dirpath, 'roads_raw_data'))
if not os.path.exists(os.path.join(roads_dirpath, 'Edges')):
    os.makedirs(os.path.join(roads_dirpath, 'Edges'))

county_mapping_filepath = os.path.join(utils.ESRD_ROOT, 'Data/County_Shapefiles/county_mapping.txt')
backup_county_mapping = {}
with open(county_mapping_filepath, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # The first part is geoid
        geoid = parts[0]
        # The rest is the county or state name (could contain spaces), ignore anything inside parentheses/comment after
        name_raw = ' '.join(parts[1:])
        # Remove anything after an open parenthesis or a '#' (comments/extra info)
        for sep in ['(', '#']:
            if sep in name_raw:
                name_raw = name_raw.split(sep)[0]
        # Remove excess whitespace
        name = name_raw.strip()
        if geoid.isdigit():
            backup_county_mapping[geoid] = name


cb_road_nodes_populations_dirpath = os.path.join(roads_dirpath, 'cb_road_nodes_populations')
if not os.path.exists(cb_road_nodes_populations_dirpath):
    os.makedirs(cb_road_nodes_populations_dirpath)
cb_road_nodes_populations_filepath_template = os.path.join(cb_road_nodes_populations_dirpath, '{}.csv')

def setup_logger_for_run(logfile):
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    root_logger = logging.getLogger()   # ROOT logger
    root_logger.setLevel(logging.INFO)
    # Remove old handlers to avoid duplication
    root_logger.handlers.clear()
    handler = logging.FileHandler(logfile, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    return root_logger


data_dirpath = os.path.join(utils.ESRD_ROOT, 'Data')
external_ssd_path = os.path.join(utils.EXTERNAL_SSD_ROOT, 'Data')

geographic_crs ='EPSG:4326'
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

counties = counties.merge(states[['STATEFP', 'STUSPS', 'STNAME']], on='STATEFP')
county_mapping = dict(zip(counties.GEOID, counties.NAME))

floodmap_dirpath = os.path.join(data_dirpath, 'uniform_pipeline_v3/FEMA_FloodMaps')
if not os.path.exists(floodmap_dirpath):
    os.makedirs(floodmap_dirpath)
simplified_floodmap_dirpath = os.path.join(data_dirpath, 'uniform_pipeline_v3/Simplified_FEMA_FloodMaps')
if not os.path.exists(simplified_floodmap_dirpath):
    # logger.info(f'Create directory: {simplified_floodmap_dirpath}')
    os.mkdir(simplified_floodmap_dirpath)

edges_dirpath = os.path.join(external_ssd_path, 'uniform_pipeline_v3/Roads/Edges')
if not os.path.exists(edges_dirpath):
    os.makedirs(edges_dirpath)


cms_dialysis_filepath = os.path.join(utils.ESRD_ROOT, 'Data/CMS_Dialysis_Locations/DFC_FACILITY_arcgis_2024.geojson')
dec_cb_census_link = 'https://api.census.gov/data/2020/dec/pl?get={}&for=block:*&in=state:{}%20county:{}%20tract:*&key={}'

county_adjacency_url = 'https://www2.census.gov/geo/docs/reference/county_adjacency.txt'
mapping_df_filepath = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/tab20_blkgrp20_blkgrp10_natl.txt')


if not os.path.join(external_ssd_path, 'uniform_pipeline_v3/baseline_conditions_node_level'):
    os.makedirs(os.path.join(external_ssd_path, 'uniform_pipeline_v3/baseline_conditions_node_level'))

db_dirpath_template = os.path.join(external_ssd_path, 'uniform_pipeline_v3/baseline_conditions_node_level/{}')
db_filepath_template = os.path.join(external_ssd_path, 'uniform_pipeline_v3/baseline_conditions_node_level/{}/esrd.db')
missing_floodmaps_filepath = os.path.join(data_dirpath, 'uniform_pipeline_v3/FEMA_FloodMaps/missing_floodmaps.pkl')
missing_floodmaps = set()
if os.path.exists(missing_floodmaps_filepath):
    with open(missing_floodmaps_filepath, 'rb') as pkl:
        missing_floodmaps = pickle.load(pkl)

empty_polygons_filepath = os.path.join(data_dirpath, 'uniform_pipeline_v3/Roads/roads_raw_data/empty_polygons.pkl')
empty_polygons = set()
if os.path.exists(empty_polygons_filepath):
    with open(empty_polygons_filepath, 'rb') as pkl:
        empty_polygons = pickle.load(pkl)

if os.path.exists(mapping_df_filepath):
    mapping_df = pd.read_csv(mapping_df_filepath, sep='|')
else:
    mapping_df = pd.read_csv('https://www2.census.gov/geo/docs/maps-data/data/rel2020/blkgrp/tab20_blkgrp20_blkgrp10_natl.txt', sep='|')

mapping_df['GEOID_BLKGRP_10'] = mapping_df.GEOID_BLKGRP_10.astype(str).str.zfill(12)
mapping_df['GEOID_BLKGRP_20'] = mapping_df.GEOID_BLKGRP_20.astype(str).str.zfill(12)
cbg_mapping = {}

for _, row in mapping_df.iterrows():
    old_cbg = row['GEOID_BLKGRP_10'][:12]
    new_cbg = row['GEOID_BLKGRP_20'][:12]
    if not old_cbg in cbg_mapping:
        cbg_mapping[old_cbg] = []
    cbg_mapping[old_cbg].append(new_cbg)

variables = [
    "P1_001N",  # Total population
    "P1_003N",  # White alone
    "P1_004N",  # Black or African American alone
    "P1_005N",  # American Indian and Alaska Native alone
    "P1_006N",  # Asian alone
    "P1_007N",  # Native Hawaiian and Other Pacific Islander alone
    "P1_008N",  # Some other race alone
    "P1_009N",   # Two or more races
    "P2_002N",   # Hispanic or Latino
    "P2_011N",    # Not Hispanic or Latino. Two or more races.
]

bridges_dirpath = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/Roads/Bridges')
if not os.path.exists(bridges_dirpath):
    os.makedirs(bridges_dirpath)
 
cms = CMSDialysisExtraction(cms_dialysis_filepath=cms_dialysis_filepath, projection=geographic_crs)

# logger.info(f'Loading nationwide CMS locations')
complete_cms_df = cms.load_cms()

# logger.info('Loading nationwide Jan to June Weekly walking patterns')
sg = SafeGraph(safegraph_dirpath=safegraph_dirpath)
complete_safegraph = sg.load_safegraph()

dialysis_nearest_nodes_dirpath = os.path.join(external_ssd_path, 'uniform_pipeline_v3/dialysis_nearest_nodes')
if not os.path.exists(dialysis_nearest_nodes_dirpath):
    os.makedirs(dialysis_nearest_nodes_dirpath)

esrd_patients_dirpath = os.path.join(utils.ESRD_ROOT, 'Data/esrdCountyLevelData')
esrd_patients = {}

for fname in os.listdir(esrd_patients_dirpath):
    if fname.startswith('Prevalent ESRD Hemo') and fname.endswith('.csv'):
        filename = os.path.join(esrd_patients_dirpath, fname)
        df = pd.read_csv(filename, low_memory=False)
        df['FIPS'] = df.FIPS.astype(str).str.zfill(5)
        df['Prevalent Count'] = df['Prevalent Count'].astype(str).str.replace(',', '').str.replace('*', '0').astype(float)
        patients = dict(zip(df.FIPS, df['Prevalent Count']))
        esrd_patients.update(patients)

# slr_dirpath_template = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/SLR_Maps/{}')
slr_dirpath_template = os.path.join(data_dirpath, 'uniform_pipeline_v3/SLR_Maps/{}')

coastal_state_abbs = [
    'ME', 'NH', 'MA', 'RI', 'CT',  # New England Atlantic
    'NY', 'NJ', 'DE', 'MD', 'VA', 'NC', 'SC', 'GA', 'FL',  # Middle/South Atlantic/Gulf
    'AL', 'MS', 'LA', 'TX',  # Gulf
    'CA', 'OR', 'WA',  # Pacific
    'AK', 'HI'  # Non-Contiguous
]
print(len(coastal_state_abbs))


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
distance_cutoff = 100.0
time_cutoff = 7200.0

if not os.path.exists(os.path.join(utils.ESRD_ROOT, 'uniform_pipeline_v3/access_to_dialysis_facilities_2024_compact_finer_node_level_external_ssd_logs')):
    os.makedirs(os.path.join(utils.ESRD_ROOT, 'uniform_pipeline_v3/access_to_dialysis_facilities_2024_compact_finer_node_level_external_ssd_logs'))
def process_county(args, empty_polygons, missing_floodmaps, rename=False, delete_old_files=False, figure_check=False):
    i, (county_geoid, county_name, state_abb, state_name, geometry) = args
    slr_check_filepaths = [
                os.path.join(external_ssd_path, f'uniform_pipeline_v3/slr_{int(elevation)}ft_node_level/{county_geoid}/od_dist.csv.gz')
                for elevation in range(0, 11)
            ]
    check_filepaths = [
        os.path.join(external_ssd_path, f'uniform_pipeline_v3/baseline_conditions_node_level/{county_geoid}/od_median.csv'),
        os.path.join(external_ssd_path, f'uniform_pipeline_v3/baseline_conditions_node_level/{county_geoid}/od_dist.csv.gz'),
        os.path.join(external_ssd_path, f'uniform_pipeline_v3/flood_conditions_node_level/{county_geoid}/od_dist.csv.gz'),
        ]
            
    

    logfile = os.path.join(utils.ESRD_ROOT, f'uniform_pipeline_v3/access_to_dialysis_facilities_2024_compact_finer_node_level_external_ssd_logs/all_cms_{county_geoid}_good_roads.log')
    logger = setup_logger_for_run(logfile)
    logger.info(f'Starting county - {county_geoid}')
    logger.info(f'Considerinng all CMS locations during SLR and Flood analysis')
    # if all([os.path.exists(filepath) for filepath in check_filepaths]):
    #     logger.info(f'Skipping county - {county_geoid} as all files exist')
    #     return

    logger.info('\n')
    logger.info('\n')
    logger.info('#'*100)
    logger.info('#'*100)
    logger.info('#'*100)
    logger.info('\n')
    logger.info('\n')
    logger.info(f'\nWorking on county_geoid: {county_geoid}\t...\t{i+1}/{n}')
    
    # Extract the neighboring counties for the current county which are within a 100 mile radius
    logger.info('Extracting study area - counties within 100 miles')
    ne = NeighborhoodExtraction(counties, geometry, cutoff=100.0)
    study_area_geoids = ne.extract_study_area()
    logger.info(f'Study area geoids: {study_area_geoids}')
    study_area = counties[counties.GEOID.isin(study_area_geoids)]
    
    logger.info('Extracting safegraph locations within the study area')
    
    safegraph = sg.extract_safegraph_within(safegraph=complete_safegraph, study_area=study_area)
    
    logger.info('Extracting CMS Dialysis Locations')
    cms_df = cms.get_cms_within(cms_df=complete_cms_df, study_area=study_area)
    good_cms_df, good_safegraph = get_good_cms_and_walking_patterns(cms_df=cms_df, 
                                                                    safegraph=safegraph[safegraph.naics_code == 621492], 
                                                                    cms_id='CMS Certification Number (CCN)', 
                                                                    sg_id='placekey',
                                                                    cutoff=.1)
    
    good_safegraph = estimate_dialysis_patients(
        cms_df = cms_df,
        safegraph = good_safegraph,
        cms_id_column = 'cms_id'
    )
    
    estimate_dialysis_patient_count = good_safegraph[['cms_id', 'estimated_dialysis_patients']].groupby('cms_id').agg({
        'estimated_dialysis_patients': 'first'
    }).estimated_dialysis_patients.sum()
    

    
    logger.info(f'Extracting od_median for the study area')
    od_median = extract_od_median(good_safegraph, cbg_mapping, county_geoid, cms_id_column='cms_id')
    od_dirpath = os.path.join(external_ssd_path, f'uniform_pipeline_v3/baseline_conditions_node_level/{county_geoid}')
    if not os.path.exists(od_dirpath):
        os.makedirs(od_dirpath)
    
    od_median_df = []
    for k, v in od_median.items():
        od_median_df.append((str(k[0]).zfill(12), int(k[1]), float(v)))
    od_median_df = pd.DataFrame(od_median_df, columns=['cbg_geoid', 'cms_id', 'median_visits'])
    od_median_df.to_csv(os.path.join(external_ssd_path, f'uniform_pipeline_v3/baseline_conditions_node_level/{county_geoid}/od_median.csv'), index=False)
    logger.info(f'OD Median saved to {os.path.join(od_dirpath, 'od_median.csv')}')

    od_median_df['county_geoid'] = od_median_df.cbg_geoid.str[:5]
    od_median_df['cms_id_int'] = od_median_df.cms_id.astype(int)
    good_cms_df['cms_id'] = good_cms_df['CMS Certification Number (CCN)'].astype(int)
    curr_state_abbs = od_median_df.merge(good_cms_df[['cms_id', 'State']], on='cms_id', how='left').State.unique()

    logger.info('Extracting Census Block Population Information')
    cb = CensusBlockInfoExtraction(
        dec_cb_census_link=dec_cb_census_link,
        dec_api_key=acs_api_key
    )

    cb_census_populations = cb.get_census_block_info(county_geoid, variables=variables, filepath_template=os.path.join(data_dirpath, 'uniform_pipeline_v3/census_block_populations/{}'))
    cb_census_populations['GEOID20'] = cb_census_populations['GEOID20'].astype(str).str.zfill(15)
    logger.info('Extracting Census Block Geometries for study area')
    cbb = CensusBlockBoundariesExtraction(
        ftp_link_template='https://www2.census.gov/geo/tiger/TIGER2020PL/LAYER/TABBLOCK/2020/tl_2020_{}_tabblock20.zip',
        save_file_template = os.path.join(data_dirpath, 'uniform_pipeline_v3/census_block_boundaries/{}')
    )
    cb_census_boundaries = cbb.get_boundaries(county_geoid)
    if cb_census_boundaries is None:
        logger.info(f'Boundaries unavailable ... Skip county geoid: {county_geoid}')
        logger.info(f'Stopping county - {county_geoid}')
        return
    
    logger.info('Extracting the road network for the study area')
    osm = OpenStreetMaps(
        study_area=study_area,
        roads_dir=roads_dirpath,
        acs_api_key=acs_api_key,
    )
    
    cb_census_boundaries['GEOID20'] = cb_census_boundaries['GEOID20'].astype(str).str.zfill(15)
    cb_census_boundaries.crs = geographic_crs

    joint_census_data = gpd.GeoDataFrame(cb_census_boundaries.merge(cb_census_populations, on='GEOID20', how='left'))
    joint_census_data.to_crs('EPSG:3857')
    joint_census_data['area'] = joint_census_data.geometry.area
    total_area = round(joint_census_data.area.sum(), 2)
    populated_area = round(joint_census_data.loc[(joint_census_data.P1_001N.notna()) & (joint_census_data.P1_001N > 0)].area.sum(), 2)
    populated_percentage = round(populated_area / total_area, 2)
    logger.info(f'Populated area in county {county_geoid}: {populated_area} / {total_area} = {populated_percentage}%')

    logger.info(f'Extracting the floodmap for the study area for county: {county_geoid}')
    ffm = FemaFloodMap(study_area, floodmap_dirpath, simplified_floodmap_dirpath, projection=geographic_crs)
    floodmap, floodmaps_missing_online = ffm.get_floodmap(missing_floodmaps)
    risky_zones = ['AE', 'A', 'AO', 'AH', 'VE', 'V']
    unknown_zones = ['AREA NOT INCLUDED', 'D']
    od_dirpath_during_flood = os.path.join(external_ssd_path, f'uniform_pipeline_v3/flood_conditions_node_level/{county_geoid}')
    if not os.path.exists(od_dirpath_during_flood):
        os.makedirs(od_dirpath_during_flood)
    
    if figure_check:
        fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
        study_area.plot(color='lightgray', edgecolor='black', linewidth=0.1, ax=ax)
        if floodmap is not None:
            if floodmap[floodmap.FLD_ZONE == 'OPEN WATER'].shape[0] > 0:
                floodmap[floodmap.FLD_ZONE == 'OPEN WATER'].plot(color='green', ax=ax)
            if floodmap[floodmap.FLD_ZONE.isin(risky_zones)].shape[0] > 0:
                floodmap[floodmap.FLD_ZONE.isin(risky_zones)].plot(color='red', ax=ax)
            if floodmap[floodmap.FLD_ZONE.isin(unknown_zones)].shape[0] > 0:
                floodmap[floodmap.FLD_ZONE.isin(unknown_zones)].plot(color='blue', ax=ax)
        # INSERT_YOUR_CODE
        import matplotlib.patches as mpatches
        legend_handles = [
            mpatches.Patch(color='lightgray', label='County boundary'),
            mpatches.Patch(color='green', label='Open Water'),
            mpatches.Patch(color='red', label='Flood Risk Zones (AE, A, AO, AH, VE, V)'),
            mpatches.Patch(color='blue', label='Unknown Zones (AREA NOT INCLUDED, D)'),
        ]
        # Add legend next to the figure
        ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, borderaxespad=0.)

        ax.set_axis_off()
        ax.set_title(f'Floodmap Zones')
        ax.margins(0.01)
        plt.savefig(os.path.join(od_dirpath_during_flood, f'floodmap_zones.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        logging.info(f'floodmap_zones.png saved to {os.path.join(od_dirpath_during_flood, f'floodmap_zones.png')}')


    if floodmap is None:
        logger.info(f'Flood layer not available for this study area')


    missing_floodmaps = missing_floodmaps.union(floodmaps_missing_online)
    with open(missing_floodmaps_filepath, 'wb') as pkl:
        pickle.dump(missing_floodmaps, pkl)


    road_graph_dist, road_graph_time, nodes, edges, empty_polygons = osm.get_road_graph(county_mapping=county_mapping, backup_county_mapping=backup_county_mapping, state_mapping=state_mapping, floodmap=floodmap, empty_polygons=empty_polygons, empty_polygons_filepath=empty_polygons_filepath, populated_percentage=populated_percentage)
    if edges is None:
        logger.info(f'No roads found for county_geoid: {county_geoid}')
        logger.info(f'Skipping county - {county_geoid}')
        return
    
    #print(f'nodes: \n{nodes.head()}')
    #print(f'edges: \n{edges.head()}')
    nodes['osmid'] = nodes['osmid'].astype(int)
    edges['osmid'] = edges['osmid'].astype(int)

    
    nodes = nodes.to_crs(cb_census_boundaries.crs)
    cb_road_nodes = gpd.sjoin(
        nodes[['osmid', 'geometry']],
        cb_census_boundaries[['GEOID20', 'geometry']],
        predicate='within',
        how='left'
    )
    #print(f'cb_road_nodes: \n{cb_road_nodes[cb_road_nodes.GEOID20.notna()].head()}')
    cb_road_nodes = gpd.GeoDataFrame(cb_road_nodes[['osmid', 'GEOID20', 'geometry']])
    cb_road_nodes_populations = cb_road_nodes.merge(cb_census_populations, on='GEOID20', how='inner')
    #print(f'cb_road_nodes_populations {county_geoid}: \n{cb_road_nodes_populations.head()}')
    # #print(f'cb_census_populations {county_geoid}: \n{cb_census_populations.head()}')

    if len(cb_road_nodes_populations) == 0:
        logger.info(f'Failed for county_geoid:{county_geoid} as no population info')
        logger.info(f'Stopping county - {county_geoid}')
        return
    
    node_counts = cb_road_nodes.groupby('GEOID20')['osmid'].count().to_dict()
    
    cb_road_nodes_populations['node_counts'] = cb_road_nodes_populations['GEOID20'].map(node_counts).fillna(0.0)
    #print(f'Before division: \n{cb_road_nodes_populations[["GEOID20", "P1_001N", "P2_002N", "node_counts"]].head()}')
    for population_column in population_columns:
        cb_road_nodes_populations[population_column] = (
                                                            cb_road_nodes_populations[population_column]
                                                            .div(cb_road_nodes_populations['node_counts'].replace(0, np.nan))
                                                            .fillna(0)
                                                        )
    #print(f'After division: \n{cb_road_nodes_populations[["GEOID20", "P1_001N", "P2_002N", "node_counts"]].head()}')
    
    cb_road_nodes_populations.drop(columns=['node_counts', 'geometry'], inplace=True)
    cb_road_nodes_populations.to_csv(cb_road_nodes_populations_filepath_template.format(county_geoid), index=False)
    cb_road_nodes_populations['cbg_geoid'] = cb_road_nodes_populations['GEOID20'].str[:12]
    zero_population_road_nodes = list(set(cb_road_nodes_populations[cb_road_nodes_populations.P1_001N == 0].osmid.unique()))

    with open(empty_polygons_filepath, 'wb') as pkl:
        pickle.dump(empty_polygons, pkl)
    
    logger.info(f'Missing Nodes: {len(set(nodes.osmid).difference(set(road_graph_dist.nodes)))} {len(set(nodes.osmid).difference(set(road_graph_time.nodes)))}')
    

    logger.info(f'Get nearest nodes to dialysis centers')
    # dialysis_nearest_nodes = get_nearest_road_nodes(nodes, good_cms_df, 'CMS Certification Number (CCN)', 'geometry', .5)
    dialysis_nearest_nodes = get_nearest_road_nodes(nodes, cms_df, 'CMS Certification Number (CCN)', 'geometry', .5)

    logger.info(f'Get nearest nodes to census block centroids')
    #print(f'dialysis_nearest_nodes: \n{list(dialysis_nearest_nodes.osmid.unique())[:10]}')
    #print(f'nodes: \n{list(nodes.osmid.unique())[:10]}')
    
    
    
    #print(f'Annotating nodes and edges for flood_100_year')
    fema_column = 'flood_100_year'    
    if not floodmap is None:
        start_time = time.time()
        if os.path.exists(os.path.join(od_dirpath_during_flood, f'flooded_nodes.pkl')):
            with open(os.path.join(od_dirpath_during_flood, f'flooded_nodes.pkl'), 'rb') as pkl:
                flooded_nodes = pickle.load(pkl)
            nodes[fema_column] = nodes.osmid.isin(flooded_nodes).astype(int)
            logger.info(f'Flooded nodes loaded from file')
        else:
            nodes = fast_annotate_nodes_strtree(nodes, floodmap.loc[floodmap.FLD_ZONE.isin(risky_zones)], fema_column)
            logger.info(f'Time taken to annotate nodes with STRTree: {time.time() - start_time} seconds')

        cb_road_nodes_merged = cb_road_nodes.merge(nodes, on='osmid')

        cb_road_nodes_pct = cb_road_nodes_merged.groupby('GEOID20')[fema_column].mean().reset_index()
        cb_road_nodes_pct = dict[Any, Any](zip(cb_road_nodes_pct.GEOID20, cb_road_nodes_pct[fema_column]))
        
        #print(f'Annotating edges for {fema_column}')
        # edges = fast_annotate_edges(edges, floodmap,  fema_column)
        start_time = time.time()
        if os.path.exists(os.path.join(od_dirpath_during_flood, f'flooded_edges.pkl')):
            with open(os.path.join(od_dirpath_during_flood, f'flooded_edges.pkl'), 'rb') as pkl:
                flooded_edges = pickle.load(pkl)
            edges[fema_column] = edges.osmid.isin(flooded_edges).astype(int)
            logger.info(f'Flooded edges loaded from file')
        else:
            edges = fast_annotate_edges_strtree(edges, floodmap.loc[floodmap.FLD_ZONE.isin(risky_zones)], fema_column, ignore_bridges=False, ignore_annot_bridges=False)
            logger.info(f'Time taken to annotate edges with STRTree: {time.time() - start_time} seconds')
        #print(f'edges: \n{edges.head()}')
    else:
        nodes[fema_column] = 0
        edges[fema_column] = 0
    
    flooded_edges = list(edges[edges[fema_column] == 1].osmid.unique())
    with open(os.path.join(od_dirpath_during_flood, f'flooded_edges.pkl'), 'wb') as pkl:
        pickle.dump(flooded_edges, pkl)
    flooded_nodes = list(nodes[nodes[fema_column] == 1].osmid.unique())
    with open(os.path.join(od_dirpath_during_flood, f'flooded_nodes.pkl'), 'wb') as pkl:
        pickle.dump(flooded_nodes, pkl)
    logger.info(f'Flooded Edges OSMID saved')

    
    
    
    num_edges_flooded = len(edges[edges[fema_column] == 1])
    #print(f'Number of edges flooded: {num_edges_flooded}')
    # edges_list = list(zip(
    #     edges[edges[fema_column] == 0]['from'],
    #     edges[edges[fema_column] == 0]['to'],
    #     edges[edges[fema_column] == 0]['osmid'],
    # ))

    od_dirpath_during_flood = os.path.join(external_ssd_path, f'uniform_pipeline_v3/flood_conditions_node_level/{county_geoid}')
    if not os.path.exists(od_dirpath_during_flood):
        os.makedirs(od_dirpath_during_flood)
    
    flooded_roads_info = {
        'scenario': 'Flood',
        'elevation': fema_column,
        'County Name': f'{county_mapping[county_geoid]} County, {state_mapping[county_geoid[:2]]}',
        'Number of Nodes': len(nodes),
        'Number of Flooded Nodes': len(nodes[nodes[fema_column] == 1]),
        'Number of Edges': len(edges),
        'Number of Flooded Edges': len(edges[edges[fema_column] == 1]),
        'Length of Edges': round(edges.length_h.sum(), 2),
        'Length of Flooded Edges': round(edges[edges[fema_column] == 1].length_h.sum(), 2),
    }
    
    with open(os.path.join(od_dirpath_during_flood, f'roads_info.json'), 'w') as f:
        json.dump(flooded_roads_info, f, indent=4)
    logger.info(f'Flooded Roads info saved to {os.path.join(od_dirpath_during_flood, f'roads_info.json')}')


    if figure_check:
        fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
        if len(study_area) > 0:
            study_area.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.1)
        if not floodmap is None and len(floodmap) > 0:
            floodmap.plot(ax=ax, color='blue', linewidth=0.1)
        if len(edges[edges[fema_column] == 1]) > 0:
            edges[edges[fema_column] == 1].plot(ax=ax, color='red', linewidth=0.25)
        if len(edges[edges[fema_column] == 0]) > 0:
            edges[edges[fema_column] == 0].plot(ax=ax, color='yellow', linewidth=0.2)
        # INSERT_YOUR_CODE
        import matplotlib.patches as mpatches
        legend_handles = [
            mpatches.Patch(color='lightgray', label='County boundary'),
            mpatches.Patch(color='blue', label='Flooded Area'),
            mpatches.Patch(color='red', label='Flooded Edges'),
            mpatches.Patch(color='yellow', label='Open Edges'),
        ]
        # Place the legend outside the right of the figure
        ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, borderaxespad=0.)
        ax.set_axis_off()
        ax.set_title(f'Roads during Flood Conditions')
        ax.margins(0.01)
        plt.savefig(os.path.join(od_dirpath_during_flood, f'roads.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
        if len(study_area) > 0:
            study_area.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.1)
        if not floodmap is None and len(floodmap) > 0:
            floodmap.plot(ax=ax, color='blue', linewidth=0.1)
        if len(edges[edges[fema_column] == 1]) > 0:
            edges[edges[fema_column] == 1].plot(ax=ax, color='red', linewidth=0.25)
    
        ax.set_axis_off()
        ax.set_title(f'Roads during Flood Conditions')
        ax.margins(0.01)
        plt.savefig(os.path.join(od_dirpath_during_flood, f'roads_flooded_only.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    
    edges_list = list(zip(
        edges[edges[fema_column] == 0]['from'],
        edges[edges[fema_column] == 0]['to'],
        edges[edges[fema_column] == 0]['osmid'],
    ))

    flooded_dialysis_nodes = set(list(nodes[nodes[fema_column] == 1].osmid.unique())).intersection(set(list(dialysis_nearest_nodes.osmid.unique())))
    flooded_good_dialysis_nodes = set(road_graph_dist.nodes).difference(flooded_dialysis_nodes)
    dialysis_nearest_nodes[fema_column] = dialysis_nearest_nodes.osmid.isin(flooded_dialysis_nodes).astype(int)


    road_graph_dist_good_nodes = road_graph_dist.subgraph(flooded_good_dialysis_nodes).copy()
    road_graph_dist_during_flood = road_graph_dist_good_nodes.edge_subgraph(edges_list).copy()
    
    road_graph_time_good_nodes = road_graph_time.subgraph(flooded_good_dialysis_nodes).copy()
    road_graph_time_during_flood = road_graph_time_good_nodes.edge_subgraph(edges_list).copy()
    
    
    #print(f'flooded_dialysis_nodes: {len(flooded_dialysis_nodes)}')
    #print(f'edges_list: {len(edges_list)} \n{edges_list[:10]}')
    #print(f'road_graph_dist nodes: {len(road_graph_dist.nodes)}')
    #print(f'road_graph_dist edges: {len(road_graph_dist.edges)} \n{list(road_graph_dist.edges)[:10]}')
    #print(f'road_graph_dist_during_flood nodes: {len(road_graph_dist_during_flood.nodes)}')
    #print(f'road_graph_dist_during_flood edges: {len(road_graph_dist_during_flood.edges)}')
    
    
    # Check for base directories for od_dist and od_time
    if not os.path.exists(os.path.join(external_ssd_path, 'uniform_pipeline_v3/baseline_conditions_node_level')):
        os.makedirs(os.path.join(external_ssd_path, 'uniform_pipeline_v3/baseline_conditions_node_level'))
    if not os.path.exists(os.path.join(external_ssd_path, 'uniform_pipeline_v3/flood_conditions_node_level')):
        os.makedirs(os.path.join(external_ssd_path, 'uniform_pipeline_v3/flood_conditions_node_level'))

    od_dirpath = os.path.join(external_ssd_path, f'uniform_pipeline_v3/baseline_conditions_node_level/{county_geoid}')
    if not os.path.exists(od_dirpath):
        os.makedirs(od_dirpath)
    
    
    
    # with open(os.path.join(od_dirpath_during_flood, 'flood_100_year_pct.pkl'), 'wb') as pkl:
    #     pickle.dump(cb_road_nodes_pct, pkl)
    logger.info(f'Working on extracting od_dist and od_time for the current study area for county: {county_geoid}')
    start_time = time.time()
    
    extract_and_dump_node_level_od_matrices_scipy(
        dialysis_nearest_nodes=dialysis_nearest_nodes,
        cms_id_col='CMS Certification Number (CCN)',
        cms_osmid_col='osmid',
        zero_population_road_nodes=zero_population_road_nodes,
        od_dirpath=od_dirpath,
        roads_length=road_graph_dist,
        distance_cutoff=distance_cutoff,
        rename=rename,
        delete_old_files=delete_old_files
    )
    logger.info(f'Time taken to extract od_dist: {(time.time() - start_time)} seconds')
    
    
    if figure_check:
        fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
        if len(study_area) > 0:
            study_area.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.2)
            # Add geoid or name of the county as text on the plot for visual reference
            for idx, row in study_area.iterrows():
                centroid = row['geometry'].representative_point()
                ax.text(
                    centroid.x, centroid.y,
                    str(row['GEOID']),
                    fontsize=6,
                    color='red',
                    ha='center',
                    va='center',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'),
                    transform=ax.transData
                )
        if len(edges) > 0:
            edges.plot(ax=ax, color='yellow', linewidth=0.25)
        # INSERT_YOUR_CODE
        import matplotlib.patches as mpatches

        legend_handles = [
            mpatches.Patch(color='lightgray', label='County boundary'),
            mpatches.Patch(color='yellow', label='Roads')
        ]
        ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.25, 1.05), frameon=True)
        
        ax.set_axis_off()
        ax.set_title(f'Baseline Study Area Roads')
        ax.margins(0.01)
        plt.savefig(os.path.join(od_dirpath, f'roads.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    
    logger.info(f'Working on extracting od_dist and od_time for the current study area during flood for county: {county_geoid}')
    start_time = time.time()
    if num_edges_flooded != 0:
        extract_and_dump_node_level_od_matrices_scipy(
            dialysis_nearest_nodes, 
            cms_id_col='CMS Certification Number (CCN)', 
            cms_osmid_col='osmid',
            zero_population_road_nodes=zero_population_road_nodes,
            od_dirpath=od_dirpath_during_flood,
            roads_length=road_graph_dist_during_flood,
            distance_cutoff=distance_cutoff,
            rename=rename,
            delete_old_files=delete_old_files
        )
        logger.info(f'Time taken to extract od_dist for flood conditions: {(time.time() - start_time)} seconds')
    else:
        logger.info(f'No edges flooded for flood conditions')

    # Overlay sea level rise data with the nodes (future underwater nodes)
    slr_dirpaths = [slr_dirpath_template.format(state_abb_slr.upper()) for state_abb_slr in curr_state_abbs]
    for elevation in range(0, 11):
        elevation_str = f'slr_{int(elevation)}ft'
        dialysis_nearest_nodes[elevation_str] = 0
        od_dirpath_slr = os.path.join(external_ssd_path, f'uniform_pipeline_v3/{elevation_str}_node_level/{county_geoid}')
        if not os.path.exists(od_dirpath_slr):
            os.makedirs(od_dirpath_slr)
            
        if os.path.exists(os.path.join(od_dirpath_slr, f'slr_nodes.pkl')) and os.path.exists(os.path.join(od_dirpath_slr, f'slr_edges.pkl')):
            slr_nodes = pickle.load(open(os.path.join(od_dirpath_slr, f'slr_nodes.pkl'), 'rb'))
            slr_edges = pickle.load(open(os.path.join(od_dirpath_slr, f'slr_edges.pkl'), 'rb'))
            slr_edges = set(slr_edges)
            slr_nodes = set(slr_nodes)
            nodes[elevation_str] = nodes.osmid.isin(slr_nodes).astype(int)
            edges[elevation_str] = edges.osmid.isin(slr_edges).astype(int)
            continue
        else:
            slr_filepaths = []
            for slr_dirpath in slr_dirpaths:
                slr_filepaths.extend(list(glob.glob(os.path.join(slr_dirpath, f'*slr_{elevation}_0ft.gzip'))) + list(glob.glob(os.path.join(slr_dirpath, f'*slr_{elevation}ft.gzip'))))
            if len(slr_filepaths) == 0:
                logger.info(f'No SLR files found for elevation: {elevation_str}')
                continue
            slr_df = []
            for slr_filepath in slr_filepaths:
                logger.info(f'\t Using SLR file: {slr_filepath}')
                slr_df.append(gpd.read_parquet(slr_filepath))
            slr_df = gpd.GeoDataFrame(pd.concat(slr_df).reset_index())
            if slr_df.crs is None:
                slr_df.crs = geographic_crs
            else:
                slr_df.to_crs(geographic_crs, inplace=True)
            
            # nodes[elevation_str] = 0
            logger.info(f'Annotating nodes for SLR {elevation_str}')
            start_time = time.time()
            if os.path.exists(os.path.join(od_dirpath_slr, f'slr_nodes.pkl')):
                with open(os.path.join(od_dirpath_slr, f'slr_nodes.pkl'), 'rb') as pkl:
                    slr_nodes = pickle.load(pkl)
                slr_nodes = set(slr_nodes)
                nodes[elevation_str] = nodes.osmid.isin(slr_nodes).astype(int)
                logger.info(f'SLR nodes loaded from file')
            else:
                nodes = fast_annotate_nodes_strtree(nodes, slr_df, elevation_str)
                logger.info(f'Time taken to annotate nodes with STRTree: {time.time() - start_time} seconds')
            logger.info(f'Number of nodes annotated for SLR {elevation_str}: {len(nodes[nodes[elevation_str] == 1])} / {len(nodes)}')
            with open(os.path.join(od_dirpath_slr, f'slr_nodes.pkl'), 'wb') as pkl:
                pickle.dump(list(nodes[nodes[elevation_str] == 1].osmid.unique()), pkl)
            
            logger.info(f'Annotating edges for SLR {elevation_str}')
            start_time = time.time()
            if os.path.exists(os.path.join(od_dirpath_slr, f'slr_edges.pkl')):
                with open(os.path.join(od_dirpath_slr, f'slr_edges.pkl'), 'rb') as pkl:
                    slr_edges = pickle.load(pkl)
                slr_edges = set(slr_edges)
                edges[elevation_str] = edges.osmid.isin(slr_edges).astype(int)
                logger.info(f'SLR edges loaded from file')
            else:
                edges = fast_annotate_edges_strtree(edges, slr_df, elevation_str, ignore_bridges=False, ignore_annot_bridges=False)
                logger.info(f'Time taken to annotate edges with STRTree: {time.time() - start_time} seconds')
            logger.info(f'Number of edges annotated for SLR {elevation_str}: {len(edges[edges[elevation_str] == 1])} / {len(edges)}')
            
            slr_edges = list(edges[edges[elevation_str] == 1].osmid.unique())
            with open(os.path.join(od_dirpath_slr, f'slr_edges.pkl'), 'wb') as pkl:
                pickle.dump(slr_edges, pkl)
            logger.info(f'SLR Edges OSMID saved')
        
        logger.info(f'SLR Nodes OSMID saved')
        cb_road_nodes_merged = cb_road_nodes.merge(nodes, on='osmid')
        cb_road_nodes_pct = cb_road_nodes_merged.groupby('GEOID20')[elevation_str].mean().reset_index()
        cb_road_nodes_pct_dict = dict(zip(cb_road_nodes_pct.GEOID20, cb_road_nodes_pct[elevation_str]))

        slr_roads_info = {
            'scenario': 'SLR',
            'elevation': elevation_str,
            'County Name': f'{county_mapping[county_geoid]} County, {state_mapping[county_geoid[:2]]}',
            'Number of Nodes': len(nodes),
            'Number of Flooded Nodes': len(nodes[nodes[elevation_str] == 1]),
            'Number of Edges': len(edges),
            'Number of Flooded Edges': len(edges[edges[elevation_str] == 1]),
            'Length of Edges': round(edges.length_h.sum(), 2),
            'Length of Flooded Edges': round(edges[edges[elevation_str] == 1].length_h.sum(), 2),
        }
        
        with open(os.path.join(od_dirpath_slr, f'roads_info.json'), 'w') as f:
            json.dump(slr_roads_info, f, indent=4)
        logger.info(f'SLR Roads info saved to {os.path.join(od_dirpath_during_flood, f'roads_info.json')}')

    
        if figure_check:
            fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
            if len(study_area) > 0:
                study_area.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.1)
            if len(slr_df) > 0:
                slr_df.plot(ax=ax, color='blue', linewidth=0.1)
            if len(edges[edges[elevation_str] == 1]) > 0:
                edges[edges[elevation_str] == 1].plot(ax=ax, color='red', linewidth=0.25)
            if len(edges[edges[elevation_str] == 0]) > 0:
                edges[edges[elevation_str] == 0].plot(ax=ax, color='yellow', linewidth=0.2)

            # INSERT_YOUR_CODE
            import matplotlib.patches as mpatches
            legend_handles = [
                mpatches.Patch(color='lightgray', label='County boundary'),
                mpatches.Patch(color='blue', label='SLR Area'),
                mpatches.Patch(color='red', label='Flooded Edges'),
                mpatches.Patch(color='yellow', label='Open Edges'),
            ]

            ax.legend(
                handles=legend_handles,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                frameon=True,
                borderaxespad=0.
            )
            ax.set_axis_off()
            ax.set_title(f'Roads during SLR {elevation_str}')
            ax.margins(0.01)
            plt.savefig(os.path.join(od_dirpath_slr, f'roads.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()


            fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
            if len(study_area) > 0:
                study_area.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.1)
            if len(slr_df) > 0:
                slr_df.plot(ax=ax, color='blue', linewidth=0.1)
            if len(edges[edges[elevation_str] == 1]) > 0:
                edges[edges[elevation_str] == 1].plot(ax=ax, color='red', linewidth=0.25)
        
            ax.set_axis_off()
            ax.set_title(f'Roads during SLR {elevation_str}')
            ax.margins(0.01)
            plt.savefig(os.path.join(od_dirpath_slr, f'roads_flooded_only.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()


        num_edges_slr = edges[elevation_str].sum()
        if num_edges_slr == 0:
            logger.info(f'No edges underwater for elevation: {elevation_str}')
            continue
        
        edges_list = list(zip(
            edges[edges[elevation_str] == 0]['from'],
            edges[edges[elevation_str] == 0]['to'],
            edges[edges[elevation_str] == 0]['osmid'],
        ))
        logger.info(f'Extracting od_dist and od_time at SLR {elevation_str}')
        slr_dialysis_nodes = set(list(nodes[nodes[elevation_str] == 1].osmid.unique())).intersection(set(list(dialysis_nearest_nodes.osmid.unique())))
        slr_good_dialysis_nodes = set(list(road_graph_dist.nodes)).difference(slr_dialysis_nodes)
        dialysis_nearest_nodes[elevation_str] = dialysis_nearest_nodes.osmid.isin(slr_dialysis_nodes).astype(int)
        
        road_graph_dist_good_nodes = road_graph_dist.subgraph(slr_good_dialysis_nodes).copy()
        road_graph_dist_slr = road_graph_dist_good_nodes.edge_subgraph(edges_list).copy()
        
        road_graph_time_good_nodes = road_graph_time.subgraph(slr_good_dialysis_nodes).copy()
        road_graph_time_slr = road_graph_time_good_nodes.edge_subgraph(edges_list).copy()
        
        logger.info(f'Running extraction of od_dist and od_time for SLR {elevation_str}')
        start_time = time.time()
        extract_and_dump_node_level_od_matrices_scipy(
            dialysis_nearest_nodes=dialysis_nearest_nodes,
            cms_id_col='CMS Certification Number (CCN)',
            cms_osmid_col='osmid',
            zero_population_road_nodes=zero_population_road_nodes,
            od_dirpath=od_dirpath_slr,
            roads_length=road_graph_dist_slr,
            distance_cutoff=distance_cutoff,
            rename=rename,
            delete_old_files=delete_old_files
        )
        logger.info(f'Time taken to extract od_dist for SLR {elevation_str}: {(time.time() - start_time)} seconds')
    dialysis_nearest_nodes = dialysis_nearest_nodes[['CMS Certification Number (CCN)', 'osmid', 'flood_100_year'] + [f'slr_{int(elevation)}ft' for elevation in range(0, 11)]]
    dialysis_nearest_nodes.to_parquet(os.path.join(dialysis_nearest_nodes_dirpath, f'{county_geoid}.parquet'))
    


from joblib import Parallel, delayed
counties['state_fips'] = counties.GEOID.str[:2].astype(int)

inputs = list(enumerate(zip(
    counties.GEOID,
    counties.NAME,
    counties.STUSPS,
    counties.STNAME,
    counties.geometry
)))

rename = False
delete_old_files = False
figure_check = True

results = Parallel(n_jobs=4, backend='loky')(
    delayed(process_county)(arg, empty_polygons, missing_floodmaps, rename=rename, delete_old_files=delete_old_files, figure_check=figure_check)
    for arg in tqdm(inputs, total=len(inputs))
)
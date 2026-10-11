from typing import Any
import geopandas as gpd
import numpy as np
import json
import numpy as np
import networkx as nx
import requests
import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
import glob
import logging
import warnings
warnings.filterwarnings("ignore")
from time import sleep
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import time
import shutil
from shapely.strtree import STRtree
logger = logging.getLogger(__name__)

# Local paths and the Census key live in the repo-root .env (not committed).
# ESRD_ROOT and EXTERNAL_SSD_ROOT are the folders that contain Data/.
load_dotenv(Path(__file__).resolve().parents[1] / '.env')
ESRD_ROOT = os.environ.get('ESRD_ROOT', '')
EXTERNAL_SSD_ROOT = os.environ.get('EXTERNAL_SSD_ROOT', '')

def get_occupied_ram():
    with open('/proc/meminfo', 'r') as meminfo:
        lines = meminfo.readlines()

    meminfo_dict = {}
    for line in lines:
        key, value = line.split(':')
        meminfo_dict[key.strip()] = int(value.split()[0])

    total_memory = meminfo_dict['MemTotal']
    free_memory = meminfo_dict['MemFree']
    available_memory = meminfo_dict['MemAvailable']
    buffers = meminfo_dict['Buffers']
    cached = meminfo_dict['Cached']
    reclaimable = meminfo_dict['SReclaimable']
    
    # Calculate used memory
    used_memory = total_memory - available_memory

    return used_memory / (1024*1024)

def annotate_nodes(nodes, floodmap, annotation_column):
    if floodmap is None:
        nodes[annotation_column] = 0
        return nodes
    # CRS match
    if floodmap.crs != nodes.crs:
        floodmap = floodmap.to_crs(nodes.crs)

    flood_bounds = floodmap.total_bounds  # [minx, miny, maxx, maxy]
    
    minx, miny, maxx, maxy = flood_bounds

    nodes_in_bbox = nodes.cx[minx:maxx, miny:maxy]

    nodes[annotation_column] = nodes.index.isin(nodes_in_bbox.index).astype(int)
    
    if nodes[annotation_column].sum() == 0:
        return nodes
    risky_nodes = nodes[nodes[annotation_column] == 1]
    safe_nodes = nodes[nodes[annotation_column] == 0]
    flooded_nodes = gpd.sjoin(
        risky_nodes,
        floodmap[['geometry']],
        predicate='within',
        how='inner'
    )
    risky_nodes[annotation_column] = risky_nodes.osmid.isin(flooded_nodes.osmid.unique()).astype(int)
    nodes = pd.concat([safe_nodes, risky_nodes])
    return nodes


def fast_annotate_nodes(nodes, floodmap, annotation_column):
    if floodmap is None:
        nodes[annotation_column] = 0
        return nodes
    spatial_index = nodes.sindex
    polygon_geoms = list(floodmap.geometry)
    indices = set()
    for i, geom in enumerate(polygon_geoms):
        index = spatial_index.query(
            geom, predicate='within',
        )
        if len(index) == 0:
            continue
        cand = nodes.geometry.take(index)
        hit_pos = cand.within(geom).to_numpy().nonzero()[0]
        if hit_pos.size == 0:
            continue
        actual_indices = nodes.index.take(index[hit_pos])
        indices.update(actual_indices)
    indices = list(indices)
    nodes[annotation_column] = 0
    nodes.loc[indices, annotation_column] = 1
    nodes.fillna(0, inplace=True)
    return nodes


def fast_annotate_edges(edges, floodmap, annotation_column):
    if floodmap is None:
        edges[annotation_column] = 0
        return edges
    
    floodmap.to_crs(edges.crs, inplace=True)
    spatial_index = edges.sindex
    polygon_geoms = list(floodmap.geometry)
    indices = set()
    for i, geom in enumerate(polygon_geoms):
        index = spatial_index.query(
            geom, predicate='intersects',
        )
        if len(index) == 0:
            continue
        # For each returned index from spatial_index.query, convert to the corresponding actual index in edges
        # Using take() is more efficient than using list indexing. List indexing can be ambiguous.
        cand = edges.geometry.take(index)
        hit_pos = cand.intersects(geom).to_numpy().nonzero()[0]
        if hit_pos.size == 0:
            continue
        actual_indices = edges.index.take(index[hit_pos])
        indices.update(actual_indices)
        # indices.update(index)
    indices = list(indices)
    edges[annotation_column] = 0
    edges.loc[indices, annotation_column] = 1
    edges[annotation_column] = edges[annotation_column].fillna(0)
    # Set annotation_column to 0 for all edges that are bridges. 0 means they cannot be flooded.
    if 'bridge' in edges.columns:
        edges['bridge_flag'] = edges.apply(lambda row: row['bridge'] != 'yes', axis=1).astype(int)
        edges[annotation_column] = edges[annotation_column] * edges['bridge_flag']
        edges.drop(columns=['bridge_flag'], inplace=True)
    return edges


# def fast_annotate_nodes_strtree(nodes, floodmap, annotation_column, predicate="within"):
#     """
#     nodes: GeoDataFrame with Point geometries
#     floodmap: GeoDataFrame with Polygon/MultiPolygon geometries
#     predicate: "intersects" (includes boundary) or "within" (strictly inside)
#     """
#     if floodmap is None or len(floodmap) == 0:
#         nodes[annotation_column] = 0
#         return nodes

#     # CRS match
#     if floodmap.crs != nodes.crs:
#         floodmap = floodmap.to_crs(nodes.crs)

#     nodes[annotation_column] = 0

#     # Build STRtree on polygons (not points)
#     polys = np.asarray(floodmap.geometry.values, dtype=object)
#     tree = STRtree(polys)

#     # Query all points; returns indices into polys for each point
#     pts = nodes.geometry.values

#     hit = np.zeros(len(nodes), dtype=np.int8)
#     for i, pt in enumerate(pts):
#         if pt is None or pt.is_empty:
#             continue
#         # candidate polygon positions that satisfy predicate
#         pos = tree.query(pt, predicate=predicate)
#         if len(pos):
#             hit[i] = 1

#     nodes[annotation_column] = hit
#     return nodes



def fast_annotate_nodes_strtree(nodes, floodmap, annotation_column):
    if floodmap is None:
        nodes[annotation_column] = 0
        return nodes
    if floodmap.crs != nodes.crs:
        floodmap.to_crs(nodes.crs, inplace=True)
    nodes[annotation_column] = 0
    node_geoms = np.asarray(nodes.geometry.values, dtype=object)
    tree = STRtree(node_geoms)
    hit_positions = set()
    for poly in floodmap.geometry.values:
        if poly is None or poly.is_empty:
            continue
        pos = tree.query(poly, predicate='intersects')
        if len(pos):
            hit_positions.update(pos.tolist())
    if hit_positions:
        hit_positions = np.fromiter(hit_positions, dtype=int)
        nodes.iloc[hit_positions, nodes.columns.get_loc(annotation_column)] = 1
    nodes[annotation_column] = nodes[annotation_column].astype('int')
    return nodes

def fast_annotate_edges_strtree(edges, floodmap, annotation_column, ignore_bridges=True, ignore_annot_bridges=True):
    edges1 = edges.copy()
    floodmap1 = floodmap.copy()
    if floodmap  is None:
        edges1[annotation_column] = 0
        return edges
    if floodmap.crs != edges1.crs:
        floodmap1.to_crs(edges1.crs, inplace=True)

    edges1[annotation_column] = 0
    # Build STRtree once
    edge_geoms = np.asarray(edges1.geometry.values, dtype=object)
    tree = STRtree(edge_geoms)

    hit_positions = set()

    # Query per polygon; predicate applied inside STRtree (exact)
    for poly in floodmap1.geometry.values:
        if poly is None or poly.is_empty:
            continue
        pos = tree.query(poly, predicate="intersects")  # returns integer positions into edge_geoms
        if len(pos):
            hit_positions.update(pos.tolist())

    if hit_positions:
        hit_positions = np.fromiter(hit_positions, dtype=int)
        edges1.iloc[hit_positions, edges1.columns.get_loc(annotation_column)] = 1

    edges1[annotation_column] = edges1[annotation_column].astype('int')
    # Set annotation_column to 0 for all edges that are bridges. 0 means they cannot be flooded.
    if ignore_bridges:
        return edges1
    if 'bridge' in edges1.columns:
        if ignore_annot_bridges:
            edges1['bridge_flag'] = edges1.apply(lambda row: row['bridge'] != 'yes' and row['bridge'] != '*', axis=1).astype(int)
            edges1[annotation_column] = edges1[annotation_column] * edges1['bridge_flag']
            edges1.drop(columns=['bridge_flag'], inplace=True)
            return edges1
        edges1['bridge_flag'] = edges1.apply(lambda row: row['bridge'] != 'yes' and row['bridge'] != '*' and row['bridge'] != 'annot_yes', axis=1).astype(int)
        edges1[annotation_column] = edges1[annotation_column] * edges1['bridge_flag']
        edges1.drop(columns=['bridge_flag'], inplace=True)
        return edges1
    return edges1




def annotate_edges(edges, floodmap, annotation_column):
    edges['osmid'] = edges['source'].astype(str) + '_' + edges['target'].astype(str)
    if floodmap is None:
        edges[annotation_column] = 0
        return edges
    flood_bounds = floodmap.total_bounds  # [minx, miny, maxx, maxy]
    minx, miny, maxx, maxy = flood_bounds
    edges_in_bbox = edges.cx[minx:maxx, miny:maxy]
    edges[annotation_column] = edges.index.isin(edges_in_bbox.index).astype(int)
    if edges[annotation_column].sum() == 0:
        return edges
    risky_edges = edges[edges[annotation_column] == 1]
    safe_edges = edges[edges[annotation_column] == 0]
    flooded_edges = gpd.sjoin(
        risky_edges,
        floodmap[['geometry']],
        predicate='intersects',
        how='inner'
    )
    risky_edges[annotation_column] = risky_edges.osmid.isin(flooded_edges.osmid).astype(int)
    edges = pd.concat([safe_edges, risky_edges])
    edges.drop(columns=['osmid'], inplace=True)
    return edges



def haversine_vec(point, locations):
        """
        Calculate the distance between two locations - (latitude, longitude)
        args:
            coord1: Point(latitude, longitude)
            coord2: Point(latitude, longitude)
        return:
            distance between two points (in miles)
        """
        # Radius of earth (in miles)
        R = 3959.0
        point = np.radians(point)
        locations = np.radians(locations)



        d = locations - point
        # dlat = lat2 - lat1
        # dlon = lon2 - lon1
        
        a = np.sin(d[:, 0]/2)**2 + np.cos(point[0]) * np.cos(locations[:, 0]) * np.sin(d[:, 1]/2)**2
        # a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        # c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = R * c
        return distance

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
def estimate_dialysis_patients(cms_df, safegraph, cms_id_column='CMS Certification Number (CCN)'):
    safegraph_copy = safegraph.copy()
    safegraph_copy.set_index(cms_id_column, inplace=True)
    estimated_dialysis_patients_dict = {}
    for curr_cms_id in safegraph_copy.index.unique():
        temp = safegraph_copy.loc[[curr_cms_id]]
        estimated_dialysis_patients1 = 0
        for idx, row in temp.iterrows():
            bucketed_dwell_times = json.loads(row['bucketed_dwell_times'])
            # dwell_time_data = bucketed_dwell_times.get('61-240', 0) + bucketed_dwell_times.get('>240', 0)
            # dwell_time_data = bucketed_dwell_times.get('>240', 0) / 3
            dwell_time_data = bucketed_dwell_times.get('>240', 0)
            estimated_dialysis_patients1 += dwell_time_data
        
        estimated_dialysis_patients1 /= len(temp)
        visitor_data = np.array([list(map(float, [i.strip('[').strip(']').strip(' ') for i in str(visits).split(',')])) for visits in temp['visits_by_day'].values])
        min_daily_visits = np.min(visitor_data, axis=0)
        estimated_dialysis_patients2 = float(np.mean(min_daily_visits))
        
        final_estimated_dialysis_patients = estimated_dialysis_patients1
        
        # cms_data = cms_df[cms_df['CMS Certification Number (CCN)'] == curr_cms_id].iloc[0]
        # # 8/2 - 4 patients per day per station
        # # divide by 3 - 3 sessions per week
        # print(f'estimated dialysis patient visitors {curr_cms_id}: {final_estimated_dialysis_patients}, {estimated_dialysis_patients2}, {cms_data["# of Dialysis Stations"]}')
        # if estimated_dialysis_patients1 > (8/2) * float(cms_data['# of Dialysis Stations']):
        #     final_estimated_dialysis_patients = min(final_estimated_dialysis_patients, estimated_dialysis_patients2)
        estimated_dialysis_patients_dict[curr_cms_id] = final_estimated_dialysis_patients
        # print(f'final estimated dialysis patient visitors {curr_cms_id}: {final_estimated_dialysis_patients}')
    safegraph['estimated_dialysis_patients'] = safegraph.cms_id.map(estimated_dialysis_patients_dict)
    return safegraph

def get_good_cms_and_walking_patterns(cms_df, safegraph, cms_id='CMS Certification Number (CCN)', sg_id='placekey', cutoff=.062371):
    safegraph_dict = {}
    for placekey, geometry in zip(safegraph[sg_id], safegraph.geometry):
        safegraph_dict[placekey] = geometry

    placekeys = list(safegraph_dict.keys())
    sg_points = np.array([np.array([coord.y, coord.x]) for coord in safegraph_dict.values()])
    placekey_to_cms_id = {}
    cms_id_to_placekey = {}
    if len(sg_points) != 0:
        for uid, cms_geom in zip(cms_df[cms_id], cms_df.geometry):
            point = np.array([cms_geom.y, cms_geom.x])
            distances = haversine_vec(point, sg_points)
            min_dist = np.min(distances)
            min_dist_idx = np.argmin(distances)
            min_placekey = placekeys[min_dist_idx]
            if min_placekey in placekey_to_cms_id:
                if min_dist < placekey_to_cms_id[min_placekey][1]:
                    placekey_to_cms_id[min_placekey] = (uid, min_dist)
            else:
                placekey_to_cms_id[min_placekey] = (uid, min_dist)
        placekey_to_cms_id = {k:v[0] for k, v in placekey_to_cms_id.items() if v[1] <= cutoff}
        cms_id_to_placekey = {v:k for k, v in placekey_to_cms_id.items()}
    

    cms_df['placekey'] = cms_df[cms_id].map(cms_id_to_placekey)
    safegraph['cms_id'] = safegraph[sg_id].map(placekey_to_cms_id)
    good_cms_df = cms_df[cms_df.placekey.notna()]
    good_safegraph = safegraph[safegraph.cms_id.notna()]
    return good_cms_df, good_safegraph

def haversine(coord1, coord2):
        """
        Calculate the distance between two locations - (latitude, longitude)
        args:
            coord1: Point(latitude, longitude)
            coord2: Point(latitude, longitude)
        return:
            distance between two points (in miles)
        """
        # Radius of earth (in miles)
        R = 3959.0
        lat1, lon1 = np.radians(coord1.y), np.radians(coord1.x)
        lat2, lon2 = np.radians(coord2.y), np.radians(coord2.x)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = R * c

        return distance

def get_nearest_road_nodes(nodes, df, uid_col, geom_col='geometry', cutoff=0.1):
    cb_nearest_nodes_dirpath = '../Data/uniform_pipeline_v3/cb_nearest_nodes'
    if not os.path.exists(cb_nearest_nodes_dirpath):
        os.makedirs(cb_nearest_nodes_dirpath)
    if uid_col == 'GEOID20':
        node_ids = list(nodes.osmid)
        coords = np.array([np.array([coord.y, coord.x]) for coord in nodes.geometry])
        uid_to_road_node = {}
        df['county_geoid'] = df.GEOID20.str[:5]
        for county_geoid in df.county_geoid.unique():
            if os.path.exists(f'../Data/uniform_pipeline_v3/cb_nearest_nodes/{county_geoid}.pkl'):
                logger.info(f'Census block nearest node exists for county: {county_geoid}')
                with open(f'../Data/uniform_pipeline_v3/cb_nearest_nodes/{county_geoid}.pkl', 'rb') as pkl:
                    sub_uid_mapping = pickle.load(pkl)
            else:
                logger.info(f'Extracting census block nearest nodes for county: {county_geoid}')
                sub_df = df[df.county_geoid == county_geoid]
                sub_uid_mapping = {}
                for uid, geom in zip(sub_df[uid_col], sub_df[geom_col]):
                    point = np.array([geom.y, geom.x])
                    distances = haversine_vec(point, coords)
                    min_dist = np.min(distances)
                    min_dist_index = np.argmin(distances)
                    if min_dist < cutoff:
                        sub_uid_mapping[uid] = node_ids[min_dist_index]
                    with open(f'../Data/uniform_pipeline_v3/cb_nearest_nodes/{county_geoid}.pkl', 'wb') as pkl:
                        pickle.dump(sub_uid_mapping, pkl)
            uid_to_road_node.update(sub_uid_mapping)
        df['osmid'] = df[uid_col].map(uid_to_road_node)
        df = df.merge(nodes[['osmid']], on='osmid')
        return df


    node_ids = list(nodes.osmid)
    coords = np.array([np.array([coord.y, coord.x]) for coord in nodes.geometry])
    uid_to_road_node = {}
    for uid, geom in zip(df[uid_col], df[geom_col]):
        point = np.array([geom.y, geom.x])
        distances = haversine_vec(point, coords)
        min_dist = np.min(distances)
        min_dist_index = np.argmin(distances)
        if min_dist < cutoff:
            uid_to_road_node[uid] = node_ids[min_dist_index]
    df['osmid'] = df[uid_col].map(uid_to_road_node)
    return df


def add_centroid(df, geographic_crs, math_crs):
    df['centroid'] = df.geometry.centroid
    df_projected = df.to_crs(math_crs)
    df['centroid'] = df_projected['centroid'].to_crs(geographic_crs)
    return df


def extract_od_median(safegraph, cbg_mapping, county_geoid, cms_id_column='CMS Certification Number (CCN)'):
    od = {}
    for _, row in safegraph.iterrows():
        poi_visits = json.loads(row['visitor_home_cbgs'])
        total_visits = sum(poi_visits.values())
        for cbg, count in poi_visits.items():
            if cbg in cbg_mapping:
                cbgs = cbg_mapping[cbg]
            else:
                cbgs = [cbg]
            for cbg1 in cbgs:
                if cbg1.startswith(county_geoid):
                    if not (cbg1, row[cms_id_column]) in od:
                        if 'estimated_dialysis_patients' in safegraph.columns:
                            od[(cbg1, row[cms_id_column])] = [count * row['estimated_dialysis_patients'] / total_visits]
                        else:
                            od[(cbg1, row[cms_id_column])] = [count]
                    else:
                        if 'estimated_dialysis_patients' in safegraph.columns:
                            od[(cbg1, row[cms_id_column])].append(count * row['estimated_dialysis_patients'] / total_visits)
                        else:
                            od[(cbg1, row[cms_id_column])].append(count)
                            
    od_median = {k:np.mean(np.array(v)) for k, v in od.items()}
    return od_median

def fetch_neighboring_counties(url):
    response = requests.get(url)
    response.raise_for_status()
    
    content = response.text
    lines = content.splitlines()
    
    neighboring_counties = {}
    current_geoid = None

    for line in lines:
        if line.strip():
            parts = line.split('\t')
            if parts[0] != '':
                current_geoid = parts[1].strip()
                neighboring_counties[current_geoid] = []
            else:
                neighboring_counties[current_geoid].append(parts[-1].strip())
    return neighboring_counties



def build_csgraph_from_nx(roads_graph, weight_attr='length_h', undirected=True):
    # 1. Fix node ordering and index mapping
    nodes = list(roads_graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    rows = []
    cols = []
    data = []

    # 2. Populate edge list
    for u, v, d in roads_graph.edges(data=True):
        w = d.get(weight_attr, 1.0)  # default weight if missing
        ui = node_to_idx[u]
        vi = node_to_idx[v]

        rows.append(ui)
        cols.append(vi)
        data.append(w)

        if undirected:
            rows.append(vi)
            cols.append(ui)
            data.append(w)

    # 3. Build CSR matrix
    A = csr_matrix((data, (rows, cols)), shape=(n, n))

    return A, nodes, node_to_idx


def csgraph_dijkstra_for_facilities(A, nodes, node_to_idx,
                                    dialysis_nearest_nodes, cms_osmid_col,
                                    distance_cutoff=np.inf):
    # Get facility node indices
    cms_nodes = dialysis_nearest_nodes[cms_osmid_col].to_numpy()
    cms_indices = [node_to_idx[n] for n in cms_nodes if n in node_to_idx]

    # Run multi-source Dijkstra
    dist_matrix = dijkstra(
        A,
        directed=False,       # or True if your graph is directed
        indices=cms_indices,  # rows correspond to these sources
        limit=distance_cutoff # like cutoff in NetworkX
    )
    # dist_matrix.shape == (n_facilities, n_nodes)
    return dist_matrix, cms_indices




def extract_and_dump_od_matrices_scipy(
        dialysis_nearest_nodes, 
        cms_id_col, 
        cms_osmid_col, 
        cb_census_populations, 
        cb_id_col, 
        cb_osmid_col, 
        od_dirpath,
        roads_length,
        roads_time,
        distance_cutoff=100.0,
        time_cutoff=7200.0
        ):
    
    od_dist_filepath = os.path.join(od_dirpath, f'od_dist.pkl')

    
    graph_nodes = list(roads_length.nodes())
    graph_adjacency_matrix = nx.to_scipy_sparse_array(roads_length, nodelist=graph_nodes, weight='length_h', format='csr')
    graph_node_to_idx = {node: i for i, node in enumerate(graph_nodes)}


    cb_node_to_geoid = dict[Any, Any](
        zip(cb_census_populations[cb_osmid_col].to_numpy(),
            cb_census_populations[cb_id_col].to_numpy())
    )

    od_dist = {}
    for cms_id, cms_node in zip(dialysis_nearest_nodes[cms_id_col], dialysis_nearest_nodes[cms_osmid_col]):
        if not cms_node in graph_nodes:
            continue
        graph_node_idx = graph_node_to_idx.get(cms_node)
        distances = dijkstra(graph_adjacency_matrix, indices=graph_node_idx)
        distances = {graph_nodes[i]:float(d) for i, d in enumerate(distances) if float(d) <= distance_cutoff}
        logger.info(f'\tNumber of distances found from {cms_id}: {len(distances)}, {distances[:5]}')
        for node, dist in distances.items():
            cb_geoid = cb_node_to_geoid.get(node)
            if cb_geoid is not None:
                od_dist[(cms_id, cb_geoid)] = dist
    with open(od_dist_filepath, 'wb') as pkl:
        pickle.dump(od_dist, pkl)




def extract_and_dump_node_level_od_matrices_scipy(
        dialysis_nearest_nodes, 
        cms_id_col,
        cms_osmid_col,
        zero_population_road_nodes,
        od_dirpath,
        roads_length,
        distance_cutoff=100.0,
        rename=False,
        delete_old_files=False,
        file_suffix=''
        ):
    od_dist_filepath = os.path.join(od_dirpath, f'od_dist_all_cms{file_suffix}.parquet')
    local_cms_id_map_filepath = os.path.join(od_dirpath, f'local_cms_id_map_all_cms{file_suffix}.pkl')
    local_node_id_map_filepath = os.path.join(od_dirpath, f'local_node_id_map_all_cms{file_suffix}.pkl')
    if os.path.exists(od_dist_filepath) and os.path.exists(local_cms_id_map_filepath) and os.path.exists(local_node_id_map_filepath):
        return
    # if rename:
    #     file_idx=0
    #     while (os.path.exists(od_dist_filepath) and os.path.exists(local_cms_id_map_filepath) and os.path.exists(local_node_id_map_filepath)):
    #         file_idx += 1
    #         od_dist_filepath = os.path.join(od_dirpath, f'od_dist_all_cms({file_idx}).parquet')
    #         local_cms_id_map_filepath = os.path.join(od_dirpath, f'local_cms_id_map_all_cms({file_idx}).pkl')
    #         local_node_id_map_filepath = os.path.join(od_dirpath, f'local_node_id_map_all_cms({file_idx}).pkl')
    
    # if delete_old_files:
    #     file_idx=0
    #     while (os.path.exists(od_dist_filepath) and os.path.exists(local_cms_id_map_filepath) and os.path.exists(local_node_id_map_filepath)):
    #         file_idx += 1
    #         od_dist_filepath = os.path.join(od_dirpath, f'od_dist({file_idx}).parquet')
    #         local_cms_id_map_filepath = os.path.join(od_dirpath, f'local_cms_id_map({file_idx}).pkl')
    #         local_node_id_map_filepath = os.path.join(od_dirpath, f'local_node_id_map({file_idx}).pkl')        
    #     if file_idx == 1:
    #         old_filepath = os.path.join(od_dirpath, f'od_dist.parquet')
    #     elif file_idx > 1:
    #         old_filepath = os.path.join(od_dirpath, f'od_dist({file_idx-1}).parquet')
    #     else:
    #         old_filepath = None
    #     if old_filepath is not None:
    #         od_dist_df = pd.read_parquet(old_filepath)
    #         if len(od_dist_df) == 0:
    #             shutil.rmtree(od_dirpath)
    #             if not os.path.exists(od_dirpath):
    #                 os.makedirs(od_dirpath)
    #                 logger.info(f'Removed od_dist and od_time for SLR {od_dirpath.split("/")[-2]} for county: {od_dirpath.split("/")[-1]}')

    #     file_idx=0
    #     while (os.path.exists(od_dist_filepath) and os.path.exists(local_cms_id_map_filepath) and os.path.exists(local_node_id_map_filepath)):
    #         file_idx += 1
    #         od_dist_filepath = os.path.join(od_dirpath, f'od_dist({file_idx}).parquet')
    #         local_cms_id_map_filepath = os.path.join(od_dirpath, f'local_cms_id_map({file_idx}).pkl')
    #         local_node_id_map_filepath = os.path.join(od_dirpath, f'local_node_id_map({file_idx}).pkl')


    if len(roads_length.nodes()) == 0:
        od_dist = pd.DataFrame(od_dist, columns=['cms_id', 'osmid', 'distance'])
        od_dist.to_parquet(od_dist_filepath, index=False)
        with open(local_cms_id_map_filepath, 'wb') as f:
            pickle.dump({}, f)
        with open(local_node_id_map_filepath, 'wb') as f:
            pickle.dump({}, f)
        return
    
    
    start_time = time.time()
    graph_nodes = list(roads_length.nodes())
    graph_adjacency_matrix = nx.to_scipy_sparse_array(roads_length, nodelist=graph_nodes, weight='length_h', format='csr')
    graph_node_to_idx = {node: i for i, node in enumerate(graph_nodes)}


    od_dist = []
    for cms_id, cms_node in zip(dialysis_nearest_nodes[cms_id_col], dialysis_nearest_nodes[cms_osmid_col]):
        if not cms_node in graph_nodes:
            continue
        graph_node_idx = graph_node_to_idx.get(cms_node)
        distances = dijkstra(graph_adjacency_matrix, indices=graph_node_idx)
        distances = {graph_nodes[i]:float(d) for i, d in enumerate(distances) if float(d) <= distance_cutoff}
        for node, dist in distances.items():
            od_dist.append((cms_id, node, dist))
    
    scenario = od_dirpath.split('/')[-2]
    logger.info(f'Time taken to extract od_dist {scenario}: {time.time() - start_time} seconds')

    # Compress the od_dist by creating local cms_id and node_id keys to reduce overhead. Use gzip compression.
    start_time = time.time()
    # od_dist = pd.DataFrame([(cms_id, node_id, distance) for (cms_id, node_id), distance in od_dist.items()], columns=['cms_id', 'osmid', 'distance'])
    od_dist = pd.DataFrame(od_dist, columns=['cms_id', 'osmid', 'distance'])
    od_dist = od_dist.loc[~od_dist.osmid.isin(zero_population_road_nodes)]
    logger.info(f'Number of {od_dirpath.split("/")[-1]} entries: {len(od_dist)}')

    unique_cms_ids = list(set(od_dist.cms_id.unique()))
    local_cms_id_map = dict(zip(unique_cms_ids, range(len(unique_cms_ids))))
    unique_node_ids = list(set(od_dist.osmid.unique()))
    local_node_id_map = dict(zip(unique_node_ids, range(len(unique_node_ids))))
    od_dist['cms_id'] = od_dist.cms_id.map(local_cms_id_map)
    od_dist['osmid'] = od_dist.osmid.map(local_node_id_map)
    od_dist['distance'] = od_dist.distance.round(2)
    logger.info(f'Time taken to compress od_dist {scenario}: {time.time() - start_time} seconds')

    start_time = time.time()
    od_dist.to_parquet(od_dist_filepath, index=False, compression='gzip')
    reverse_local_cms_id_map = {v:k for k,v in local_cms_id_map.items()}
    reverse_local_node_id_map = {v:k for k,v in local_node_id_map.items()}
    with open(local_cms_id_map_filepath, 'wb') as f:
        pickle.dump(reverse_local_cms_id_map, f)
    with open(local_node_id_map_filepath, 'wb') as f:
        pickle.dump(reverse_local_node_id_map, f)
    logger.info(f'Time taken to save od_dist {scenario}: {time.time() - start_time} seconds')

    all_distances = od_dist.distance.values
    if len(all_distances) == 0 or all_distances.sum() == 0:
        logger.info(f'No distances found')
        return
    
    logger.info(f'mean distance: {np.mean(all_distances)}')
    logger.info(f'std distance: {np.std(all_distances)}')
    logger.info(f'median distance: {np.median(all_distances)}')
    logger.info(f'max distance: {np.max(all_distances)}')
    logger.info(f'min distance: {np.min(all_distances)}')

def extract_and_dump_od_matrices(
        dialysis_nearest_nodes, 
        cms_id_col, 
        cms_osmid_col, 
        cb_census_populations, 
        cb_id_col, 
        cb_osmid_col, 
        od_dirpath,
        roads_length,
        roads_time,
        distance_cutoff=100.0,
        time_cutoff=7200.0
        ):
    od_dist = {}
    od_time = {}
    od_dist_filepath = os.path.join(od_dirpath, f'od_dist_nx.pkl')
    od_time_filepath = os.path.join(od_dirpath, f'od_time_nx.pkl')
    # if os.path.exists(od_dist_filepath) and os.path.exists(od_time_filepath):
    #     logger.info(f'od_dist and od_time already exist')
    #     return
    cb_node_to_geoid = dict[Any, Any](
        zip(cb_census_populations[cb_osmid_col].to_numpy(),
            cb_census_populations[cb_id_col].to_numpy())
    )

    for cms_id, cms_node in zip(dialysis_nearest_nodes[cms_id_col], dialysis_nearest_nodes[cms_osmid_col]):
        if not cms_node in roads_length.nodes:
            continue
        distances = nx.single_source_dijkstra_path_length(roads_length, source=cms_node, cutoff=distance_cutoff, weight='length_h')
        for node, dist in distances.items():
            cb_geoid = cb_node_to_geoid.get(node)
            if cb_geoid is not None:
                od_dist[(cms_id, cb_geoid)] = dist

        # times = nx.single_source_dijkstra_path_length(roads_time, source=cms_node, cutoff=time_cutoff, weight='time')
        # for node, time_ in times.items():
        #     cb_geoid = cb_node_to_geoid.get(node)
        #     if cb_geoid is not None:
        #         od_time[(cms_id, cb_geoid)] = time_

    with open(od_dist_filepath, 'wb') as pkl:
        pickle.dump(od_dist, pkl)
    # with open(od_time_filepath, 'wb') as pkl:
    #     pickle.dump(od_time, pkl)



# def extract_and_dump_od_matrices(
#         dialysis_nearest_nodes, 
#         cms_id_col, 
#         cms_osmid_col, 
#         cb_census_populations, 
#         cb_id_col, 
#         cb_osmid_col, 
#         od_dirpath,
#         roads_length,
#         roads_time,
#         distance_cutoff=100.0,
#         time_cutoff=7200.0
#         ):
#     od_dist = {}
#     od_time = {}
#     od_dist_filepath = os.path.join(od_dirpath, f'od_dist.pkl')
#     od_time_filepath = os.path.join(od_dirpath, f'od_time.pkl')
#     if os.path.exists(od_dist_filepath) and os.path.exists(od_time_filepath):
#             logger.info(f'File already exists for dialysis location cb_geoid: {cb_geoid}')
#             return
    
#     for cb_geoid, cb_node in zip(cb_census_populations[cb_id_col], cb_census_populations[cb_osmid_col]):
#         if not cb_node in roads_length.nodes:
#             continue

#         distances = nx.single_source_dijkstra_path_length(roads_length, source=cb_node, cutoff=distance_cutoff, weight='length_h')
#         for cms_id, cms_node in zip(dialysis_nearest_nodes[cms_id_col], dialysis_nearest_nodes[cms_osmid_col]):
#             if cms_node in distances:
#                 od_dist[(cms_id, cb_geoid)] = distances[cms_node]

#         times = nx.single_source_dijkstra_path_length(roads_time, source=cb_node, cutoff=time_cutoff, weight='time')
#         for cms_id, cms_node in zip(dialysis_nearest_nodes[cms_id_col], dialysis_nearest_nodes[cms_osmid_col]):
#             if cms_node in times:
#                 od_time[(cms_id, cb_geoid)] = times[cms_node]

#     with open(od_dist_filepath, 'wb') as pkl:
#         pickle.dump(od_dist, pkl)
#     with open(od_time_filepath, 'wb') as pkl:
#         pickle.dump(od_time, pkl)

#     for cms_id, cms_node in zip(dialysis_nearest_nodes[cms_id_col], dialysis_nearest_nodes[cms_osmid_col]):
#         if not cms_node in roads_length.nodes:
#             continue
#         od_dist_filepath = os.path.join(od_dirpath, f'od_dist_{cms_id}.pkl')
#         od_time_filepath = os.path.join(od_dirpath, f'od_time_{cms_id}.pkl')
#         if os.path.exists(od_dist_filepath) and os.path.exists(od_time_filepath):
#             logger.info(f'File already exists for dialysis location cms_id: {cms_id}')
#             continue
#         distances = nx.single_source_dijkstra_path_length(roads_length, source=cms_node, cutoff=distance_cutoff, weight='length_h')
#         od_dist = {}
#         for cb_geoid, cb_node in zip(cb_census_populations[cb_id_col], cb_census_populations[cb_osmid_col]):
#             if cb_node in distances:
#                 od_dist[(cms_id, cb_geoid)] = distances[cb_node]
        
#         times = nx.single_source_dijkstra_path_length(roads_time, source=cms_node, cutoff=time_cutoff, weight='time')
#         od_time = {}
#         for cb_geoid, cb_node in zip(cb_census_populations[cb_id_col], cb_census_populations[cb_osmid_col]):
#             if cb_node in times:
#                 od_time[(cms_id, cb_geoid)] = times[cb_node]
        
#         with open(od_dist_filepath, 'wb') as pkl:
#             pickle.dump(od_dist, pkl)
#         with open(od_time_filepath, 'wb') as pkl:
#             pickle.dump(od_time, pkl)


def remove_cached_od_files(od_dirpath):
    for filename in glob.glob(os.path.join(od_dirpath, 'od_*.pkl')):
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass

def is_file_greater_than(filepath, size_threshold):
    try:
        file_size = os.path.getsize(filepath)
        return file_size > size_threshold
    except OSError as e:
        logger.info(f"Error in checking database size: {e}")
        return False
    


def get_url_with_retry(url, timeout=30, max_retries=5, retry_delay=5):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            return response
        except requests.Timeout:
            logger.info(f"\tTimeout occurred. Retrying {attempt + 1}/{max_retries}...")
            sleep(retry_delay)
        except requests.RequestException as e:
            logger.info(f"\tA RequestException error occurred: {e}")
            break  # For other types of exceptions, break the loop and do not retry
    logger.info("\tMax retries exceeded. Failed to fetch the URL.")
    return None


# def flooding_and_isolation(
#     od_median_filepath, 
#     od_dist_filepath, 
#     cms_id_map_filepath,
#     road_nodes_map_filepath,
#     od_dist_slr_filepath, 
#     cms_id_map_slr_filepath,
#     road_nodes_map_slr_filepath,
#     cb_road_nodes_population_filepath
# ):
#     od_median_df = pd.read_csv(od_median_filepath)
#     od_median_df['cms_id'] = od_median_df['cms_id'].astype(int)
    
#     cb_road_nodes_population_df = pd.read_csv(cb_road_nodes_population_filepath)
#     cb_road_nodes_population_df['cbg_geoid'] = cb_road_nodes_population_df['cbg_geoid'].astype(str).str.zfill(12)
    
#     od_dist_df = pd.read_parquet(od_dist_filepath)
#     cms_id_map = pickle.load(open(cms_id_map_filepath, 'rb'))
#     road_nodes_map = pickle.load(open(road_nodes_map_filepath, 'rb'))
#     od_dist_df['cms_id'] = od_dist_df['cms_id'].map(cms_id_map)
#     od_dist_df['osmid'] = od_dist_df['osmid'].map(road_nodes_map)
#     od_dist_df['cms_id'] = od_dist_df['cms_id'].astype(int)
    
#     od_dist_slr_df = pd.read_parquet(od_dist_slr_filepath)
#     cms_id_map_slr = pickle.load(open(cms_id_map_slr_filepath, 'rb'))
#     road_nodes_map_slr = pickle.load(open(road_nodes_map_slr_filepath, 'rb'))
#     od_dist_slr_df['cms_id'] = od_dist_slr_df['cms_id'].map(cms_id_map_slr)
#     od_dist_slr_df['osmid'] = od_dist_slr_df['osmid'].map(road_nodes_map_slr)
#     od_dist_slr_df['cms_id'] = od_dist_slr_df['cms_id'].astype(int)
#     od_dist_slr_df.rename(columns={'distance': 'distance_slr'}, inplace=True)

#     cb_road_nodes_population_df = pd.read_parquet(cb_road_nodes_population_filepath)

#     merged_df = od_dist_df.merge(cb_road_nodes_population_df[['osmid', 'cbg_geoid', 'P1_001N']], on='osmid', how='inner')
#     merged_df = merged_df.merge(od_median_df, on=['cms_id', 'cbg_geoid'], how='inner')

#     merged_slr_df = od_dist_slr_df.merge(cb_road_nodes_population_df[['osmid', 'cbg_geoid', 'P1_001N']], on='osmid', how='inner')
#     merged_slr_df = merged_slr_df.merge(od_median_df, on=['cms_id', 'cbg_geoid'], how='inner')

#     merged_diff_df = merged_df.merge(merged_slr_df[['cms_id', 'cbg_geoid', 'distance']], on=['cms_id', 'cbg_geoid'], how='left')
#     merged_diff_df['distance_diff'] = merged_diff_df.distance_slr - merged_diff_df.distance
#     merged_diff_df.sort_values(by='distance_diff', ascending=False, inplace=True)
#     return merged_diff_df
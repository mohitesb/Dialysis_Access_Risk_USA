import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point, MultiPoint
import numpy as np
import pandas as pd
import gc
import os
from data_downloader import FemaFloodMapDownloader, OpenStreetMapDownloader, CensusBlockInfoDownloader, CensusBlockBoundariesDownloader
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import ast
import shapely.geometry as geom
import shapely.ops as ops
import matplotlib.pyplot as plt
import glob
import utils
from shapely.geometry import Point
from census import Census
import pickle
import logging
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from time import sleep
import pickle
import shutil
from shapely.geometry import LineString
from shapely import wkt
from pathlib import Path
from filelock import FileLock
import time 
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)
county_manual_mapping = {
    '02020' : 'Anchorage Municipality',
}


class CensusBlockBoundariesExtraction:
    def __init__(
            self,
            ftp_link_template='https://www2.census.gov/geo/tiger/TIGER2020PL/LAYER/TABBLOCK/2020/tl_2020_{}_tabblock20.zip',
            save_file_template = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/census_block_boundaries/{}'),
            projection='EPSG:4326'
    ):
        self.ftp_link_template = ftp_link_template
        self.save_file_template = save_file_template
        self.projection = projection

    def get_boundaries(self, county_geoid):
        cbbd = CensusBlockBoundariesDownloader(
            ftp_link_template = self.ftp_link_template,
        )
        dirname = self.save_file_template.format(county_geoid)
        if not os.path.exists(dirname) or not glob.glob(os.path.join(dirname, "*.shp")):
            os.makedirs(dirname, exist_ok=True)
            logger.info(f'Downloading for county: {county_geoid}')
            cbbd.download(county_geoid, dirname)
        cb_census_boundary = None
        for file in os.listdir(dirname):
            if file.endswith('.shp'):
                shapefile_path = os.path.join(dirname, file)
                cb_census_boundary = gpd.read_file(shapefile_path)
                cb_census_boundary['GEOID20'] = cb_census_boundary.GEOID20.astype(str).str.zfill(15)
                cb_census_boundary.crs = self.projection
        del cbbd
        return cb_census_boundary

class CensusBlockInfoExtraction:
    def __init__ (
            self,
            dec_cb_census_link = 'https://api.census.gov/data/2020/dec/pl?get={}&for=block:*&in=state:{}%20county:{}%20tract:*&key={}',
            dec_api_key = os.environ['CENSUS_API_KEY'],
    ):
        self.dec_cb_census_link = dec_cb_census_link
        self.dec_api_key = dec_api_key

    
    def get_census_block_info(self, county_geoid, variables, filepath_template = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3s/census_block_populations/{}')):
        cbid = CensusBlockInfoDownloader(
            dec_cb_census_link=self.dec_cb_census_link,
            dec_api_key=self.dec_api_key
        )
        filepath = filepath_template.format(f'{county_geoid}.csv')
        state_geocode = county_geoid[:2]
        county_geocode = county_geoid[2:]
        if not os.path.exists(filepath):
            logger.info(f'Downloading for county: {county_geoid}')
            cb_census = None
            num_requests = 0
            while cb_census is None and num_requests < 4:
                cb_census = cbid.download(variables, state_geocode, county_geocode, filepath)
                num_requests += 1
        cb_census = pd.DataFrame(['GEOID20'] + variables)
        if os.path.exists(filepath):
            cb_census = pd.read_csv(filepath)
            cb_census['state'] = cb_census.state.astype(str).str.zfill(2)
            cb_census['county'] = cb_census.county.astype(str).str.zfill(3)
            cb_census['tract'] = cb_census.tract.astype(str).str.zfill(6)
            cb_census['block'] = cb_census.block.astype(str).str.zfill(4)
            cb_census['GEOID20'] = cb_census['state'] + cb_census['county'] + cb_census['tract'] + cb_census['block']
            cb_census = cb_census[['GEOID20'] + variables]
        del cbid
        return cb_census






class CMSDialysisExtraction:
    def __init__ (
            self,
            cms_dialysis_filepath = os.path.join(utils.ESRD_ROOT, 'Data/CMS_Dialysis_Locations/DFC_FACILITY_missing_merged.geojson'),
            projection='EPSG:4326'
    ):
        self.cms_dialysis_filepath = cms_dialysis_filepath
        self.projection = projection
    
    def load_cms(self):
        cms_df = gpd.read_file(self.cms_dialysis_filepath)
        return cms_df

    def get_cms_within(self, cms_df, study_area):
        if cms_df.crs is None:
            cms_df.crs = self.projection
        else:
            cms_df.to_crs(self.projection, inplace=True)
        cms_df = gpd.sjoin(cms_df, study_area, predicate='within')        
        return cms_df



class NeighborhoodExtraction:
    def __init__ (self, counties, geometry, cutoff, projection='EPSG:4326'):
        self.counties = counties
        self.geometry = geometry
        self.cutoff = cutoff
        self.projection = projection

    def _create_circle(self, point, radius):
        return point.buffer(radius)

    def extract_study_area(self):
        if isinstance(self.geometry, MultiPolygon):
            polygons = self.geometry.geoms
        elif isinstance(self.geometry, Polygon):
            polygons = [self.geometry]
        else:
            raise ValueError("Input geometry must be a MultiPolygon or Polygon")
        circles = []
        for polygon in polygons:
            for point in polygon.exterior.coords:
                shapely_point = Point(point)
                # Convert miles to degrees (approximately)
                circle = self._create_circle(shapely_point, self.cutoff / (111.32 * np.cos(np.radians(shapely_point.y))))
                circles.append(circle)
        circles = gpd.GeoDataFrame(circles, columns=['geometry'])
        circles.crs = self.projection
        neighboring = gpd.sjoin(circles, self.counties[['GEOID', 'geometry']], predicate='intersects')
        neighboring_geoids = list(set(neighboring.GEOID))
        return neighboring_geoids
    
class SafeGraph:
    def __init__(self, safegraph_dirpath=None, safegraph_filepath=None, projection='EPSG:4326'):
        self.safegraph_dirpath = safegraph_dirpath
        self.safegraph_filepath = safegraph_filepath
        self.projection = projection

    def _add_geometry(self, safegraph):
        safegraph['geometry'] = [Point(longitude, latitude) for longitude, latitude in zip(safegraph.longitude, safegraph.latitude)]
        safegraph = gpd.GeoDataFrame(safegraph)
        safegraph.crs = self.projection
        return safegraph
    
    def load_safegraph(self):
        if self.safegraph_filepath is not None:
            safegraph = pd.read_csv(self.safegraph_filepath)
        elif self.safegraph_dirpath is not None:
            safegraph = []
            for fname in os.listdir(self.safegraph_dirpath):
                if not fname.endswith('.csv.gz'):
                    continue
                filename = os.path.join(self.safegraph_dirpath, fname)
                try:
                    df = pd.read_csv(filename, compression='gzip')
                    if df.empty:
                        logger.info(f'Skipping empty file: {fname}')
                    safegraph.append(df[df.visitor_home_cbgs.notna()].loc[:])
                except pd.errors.EmptyDataError:
                    logger.info(f'EmptyDataEror: No columns to parse from file: {fname}')
                    continue
            safegraph = pd.concat(safegraph)
        else:
            raise Exception('Provide atleast one - safegraph_filepath or safegraph_dirpath')
        safegraph = self._add_geometry(safegraph=safegraph)
        return safegraph
    
    def extract_safegraph_within(self, safegraph, study_area):
        safegraph = gpd.sjoin(safegraph, study_area[['GEOID', 'geometry']], predicate='within')
        return safegraph


class FemaFloodMap:
    def __init__(self, study_area, floodmap_dirpath, simplified_floodmap_dirpath, projection ='EPSG:4326'):
        self.study_area = study_area
        self.floodmap_dirpath = floodmap_dirpath
        self.simplified_floodmap_dirpath = simplified_floodmap_dirpath
        self.projection = projection
    def _check_if_simplfied_floodmap_exists(self, geoid):
        return any([os.path.exists(os.path.join(self.simplified_floodmap_dirpath, dirname, 'S_FLD_HAZ_AR.shp')) for dirname in os.listdir(self.simplified_floodmap_dirpath)])

    def _check_if_floodmap_exists(self, geoid):
        return any([dirname.startswith(geoid) for dirname in os.listdir(self.floodmap_dirpath)])
    
    def get_floodmap(self, missing_floodmaps):
        floodmap_missing_online = set()
        missing_geoids = set()
        for geoid in self.study_area.GEOID:
            if geoid in missing_floodmaps:
                logger.info(f'\t {geoid} -Skipping ... Floodmap recognized as missing floodmap on FEMA website')
                continue
            if not self._check_if_simplfied_floodmap_exists(geoid):
                if not self._check_if_floodmap_exists(geoid):
                    missing_geoids.add(geoid)

        if missing_geoids:
            logger.info(f'Missing geoids: {sorted(missing_geoids)}')
            femaFloodMapDownloader = FemaFloodMapDownloader(
            county_geoids=missing_geoids,
            wait_time=30,
            link = 'https://hazards.fema.gov/femaportal/NFHL/searchResult'
            )
            femaFloodMapDownloader.download()
            del femaFloodMapDownloader
            for geoid in missing_geoids:
                if not self._exists(geoid):
                    floodmap_missing_online.add(geoid)

        for geoid in missing_geoids:
            for dirname in os.listdir(self.floodmap_dirpath):
                if dirname.startswith(geoid):
                    filename = os.path.join(self.floodmap_dirpath, dirname, 'S_FLD_HAZ_AR.shp')
                    if os.path.exists(filename):
                        curr_floodmap = gpd.read_file(filename)
                        if not os.path.exists(os.path.join(self.simplified_floodmap_dirpath, dirname)):
                            os.mkdir(os.path.join(self.simplified_floodmap_dirpath, dirname))
                        logger.info(f'\tCreate simplified floodmap shapefile for geoid: {geoid}')
                        simplified_filename = os.path.join(self.simplified_floodmap_dirpath, dirname, 'S_FLD_HAZ_AR.shp')
                        curr_floodmap[curr_floodmap.FLD_ZONE != 'X'][['FLD_ZONE', 'geometry']].to_file(simplified_filename)
                        logger.info(f'\tDelete the temporary bulky floodmap file: {geoid}')
                        shutil.rmtree(os.path.join(self.floodmap_dirpath, dirname))
        
        logger.info('\tExtracting simplified floodmap for the study area')
        floodmaps = []
        for geoid in self.study_area.GEOID:
            for dirname in os.listdir(self.simplified_floodmap_dirpath):
                if dirname.startswith(geoid):
                    filename = os.path.join(self.simplified_floodmap_dirpath, dirname, 'S_FLD_HAZ_AR.shp')
                    curr_floodmap = gpd.read_file(filename)
                    curr_floodmap['GEOID'] = str(geoid).zfill(5)
                    if curr_floodmap.crs is None:
                        curr_floodmap.crs = self.projection
                    else:
                        curr_floodmap.to_crs(self.projection, inplace=True)
                    floodmaps.append(curr_floodmap)
        if floodmaps:
            # res_floodmaps = pd.concat(floodmaps, axis=0)
            res_floodmaps_filtered = gpd.GeoDataFrame(pd.concat(floodmaps), geometry='geometry', crs=self.projection)
            # res_floodmaps_filtered = res_floodmaps.loc[res_floodmaps.FLD_ZONE != 'X'].copy()
            return res_floodmaps_filtered, floodmap_missing_online
        else:
            return None, floodmap_missing_online

class OpenStreetMaps:
    def __init__ (
            self,
            study_area,
            roads_dir = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/Roads'),
            acs_api_key=None,
            projection = 'EPSG:4326'
    ):
        self.study_area = study_area
        self.roads_dir = roads_dir
        self.acs_api_key = os.environ['CENSUS_API_KEY'] if acs_api_key is None else acs_api_key
        self.projection = projection
        
        states = pd.read_csv(os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/fips_to_statename.txt'), sep='        ', header=None, engine='python')
        states.columns = ['FIPS', 'state']
        states['FIPS'] = states.FIPS.astype(str).str.zfill(2)
        # logger.info(states)
        self.states = dict(zip(
            states.FIPS, states.state
        ))
        self.census = Census(self.acs_api_key)
        self.osm_downloader = OpenStreetMapDownloader()
    

    def maxspeed_to_float(self, edges):
        maxspeeds = []
        for maxspeed in edges.maxspeed.str.lower():
            if maxspeed.startswith('['):
                temp = ast.literal_eval(maxspeed)
                maxspeeds.append(np.mean(np.array([float(i.split(' ')[0]) for i in temp if i != 'unposted' and i != 'signals' if  i.split(' ')[0]!='' and all(chr.isdigit() or chr == '.' for chr in i.split(' ')[0])])))
            elif '-' in maxspeed:
                temp_speeds = [float(i.strip()) for i in maxspeed.split('-') if i!='' and all(chr.isdigit() or chr == '.' for chr in i)]
                if temp_speeds:
                    maxspeeds.append(sum(temp_speeds) / len(temp_speeds))
                else:
                    maxspeeds.append(0)
            elif ',' in maxspeed:
                temp_speeds = [float(i.strip()) for i in maxspeed.split(',') if i!='' and all(chr.isdigit() or chr == '.' for chr in i)]
                if temp_speeds:
                    maxspeeds.append(sum(temp_speeds) / len(temp_speeds))
                else:
                    maxspeeds.append(0)
            elif maxspeed == '0' or maxspeed.endswith(' mph') or maxspeed.endswith(' mphzz'):
                maxspeeds.append(float(maxspeed.split(' ')[0]))
            elif maxspeed == 'unposted' or maxspeed == 'signals' or maxspeed == 'none' or 'implicit' in maxspeed or 'default' in maxspeed or '/' in maxspeed:
                maxspeeds.append(0)
            elif maxspeed.startswith('posted') or maxspeed.startswith('unsigned') or maxspeed.startswith('signed'):
                try:
                    maxspeeds.append(float(maxspeed.split(' ')[1].strip()))
                except:
                    maxspeeds.append(0.0)
            elif ';' in maxspeed or '@' in maxspeed:
                maxspeeds.append(float(maxspeed.split(' ')[0]))
            elif 'slow' in maxspeed:
                maxspeeds.append(10.0)
            elif maxspeed.endswith('mph') or maxspeed.endswith(' mphzz'):
                maxspeeds.append(float(maxspeed[:-3]))
            else:
                try:
                    maxspeeds.append(float(maxspeed))
                except:
                    maxspeeds.append(25.0)
                    # logger.error(maxspeed, len(maxspeed))
        return maxspeeds


    def map_maxspeeds(self, data):
        maxspeeds = {}
        for _, row in data.iterrows():
            if not row['highway'] in maxspeeds:
                maxspeeds[row['highway']] = []
            maxspeeds[row['highway']].append(row['maxspeed'])
        # max value is 25 to ensure minimum speed in the data is 25mph
        maxspeeds = {
            k: max(25, np.mean(np.array(v))) for k, v in maxspeeds.items()
        }
        return maxspeeds

    def encode_highway(self, highways):
        label_encoder = LabelEncoder()
        return label_encoder.fit_transform(highways)

    def estimate_maxspeeds(self, edges, maxspeeds):
        estimated_maxspeed = []
        for _, row in edges.iterrows():
            if row['maxspeed'] < 10.0:
                if row['highway'] in maxspeeds:
                    estimated_maxspeed.append(maxspeeds[row['highway']])
                else:
                    # Assume missing data as 30 mph
                    estimated_maxspeed.append(30)
            else:
                estimated_maxspeed.append(row['maxspeed'])
        return estimated_maxspeed
    


    def haversine(self, coord1, coord2):
        R = 3959.0  # Earth's radius in kilometers

        lat1, lon1 = np.radians(coord1[0]), np.radians(coord1[1])
        lat2, lon2 = np.radians(coord2[0]), np.radians(coord2[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = R * c

        return distance

    def get_road_lengths(self, edges):
        lengths = []
        for linestring in edges:
            coords = list(linestring.coords)
            length = 0
            for i in range(1, len(coords)):
                length += self.haversine(coords[i-1], coords[i])
            lengths.append(length)
        return lengths
    
    def build_graph(self, nodes, edges):
        roads_graph_time = nx.from_pandas_edgelist(edges, 'from', 'to', edge_attr='time', edge_key='osmid', create_using=nx.MultiGraph())
        roads_graph_dist = nx.from_pandas_edgelist(edges, 'from', 'to', edge_attr='length_h', edge_key='osmid', create_using=nx.MultiGraph())
        return roads_graph_dist, roads_graph_time
    
    def build_distance_graph(self, nodes, edges):
        roads_graph_dist = nx.from_pandas_edgelist(edges, 'from', 'to', edge_attr='length_h', edge_key='osmid', create_using=nx.MultiGraph())
        return roads_graph_dist

    def create_sub_polygon(self, union_polygon, startx, endx, miny, maxy):
        bbox = geom.box(startx, miny, endx, maxy)
        return union_polygon.intersection(bbox)
    
    def clockwise_sort(self, points):
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:,1] - centroid[1], points[:,0] - centroid[0])
        sorted_indices = np.argsort(angles)
        return points[sorted_indices]

    def write_parquet_with_lock(self, data, save_filename):
        dst = Path(save_filename)
        lock = FileLock(str(dst) + '.lock')
        with lock:
            if dst.exists() and dst.stat().st_size > 0:
                return str(dst)
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            data.to_parquet(tmp, index=False)
            os.replace(tmp, dst)
            return 'success'
    
    def write_shapefile_with_lock(self, data, save_filename):
        dst = Path(save_filename)
        lock = FileLock(str(dst) + '.lock')
        with lock:
            if dst.exists() and dst.stat().st_size > 0:
                return str(dst)
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            data.to_file(tmp, index=False, driver='ESRI Shapefile')
            os.replace(tmp, dst)
            return 'success'

    def write_pickle_with_lock(self, data, save_filename):
        dst = Path(save_filename)
        lock = FileLock(str(dst) + '.lock')
        with lock:
            if dst.exists() and dst.stat().st_size > 0:
                return str(dst)
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            with open(tmp, 'wb') as pkl:
                pickle.dump(data, pkl)
            os.replace(tmp, dst)
            return 'success'
    
    def explode_osmid_column(self, df):
        # If 'osmid' is a list or string representation of a list, expand it
        def to_list(x):
            if isinstance(x, list):
                return x
            elif isinstance(x, str) and x.startswith('[') and x.endswith(']'):
                try:
                    return ast.literal_eval(x)
                except Exception:
                    return [x]
            else:
                return [x]
        
        df = df.copy()
        df['osmid'] = df['osmid'].apply(to_list)
        df = df.explode('osmid', ignore_index=True)
        return df
    
    def extract_and_dump_tiger(self, nodes_filename, edges_filename, road_graph_dist_filename):
        nodes = gpd.read_file(nodes_filename)
        if nodes.crs is None:
            nodes.crs = self.projection
        else:
            nodes.to_crs(self.projection, inplace=True)
        nodes['osmid'] = nodes['osmid'].astype(int)
        
        edges = gpd.read_file(edges_filename)
        if edges.crs is None:
            edges.crs = self.projection
        else:
            edges.to_crs(self.projection, inplace=True)
        edges['osmid'] = edges['osmid'].astype(int)
        
        road_graph_dist = self.build_distance_graph(nodes, edges)
        self.write_pickle_with_lock(road_graph_dist, road_graph_dist_filename)
    
    def extract_and_dump(self, nodes_filename, edges_filename, road_graph_dist_filename, road_graph_time_filename):
        nodes_gdf = gpd.read_file(nodes_filename)       
        edges_gdf = gpd.read_file(edges_filename)
        edges_gdf = self.explode_osmid_column(edges_gdf)
        edges_gdf['osmid'] = edges_gdf['osmid'].astype(int)
        nodes_gdf['osmid'] = nodes_gdf['osmid'].astype(int)

        if nodes_gdf.crs is None:
            nodes_gdf.crs = self.projection
        else:
            nodes_gdf.to_crs(self.projection, inplace=True)
        if edges_gdf.crs is None:
            edges_gdf.crs = self.projection
        else:
            edges_gdf.to_crs(self.projection, inplace=True)
        edges_gdf.crs = self.projection

        # print(f'edges_gdf columns: \n{edges_gdf.columns}')

        road_graph_dist, road_graph_time = self.build_graph(nodes_gdf, edges_gdf)
        self.write_pickle_with_lock(road_graph_dist, road_graph_dist_filename)
        self.write_pickle_with_lock(road_graph_time, road_graph_time_filename)


    def extract_adjacent_roads(self, geoid, adj_geoid, nodes_1, nodes_2, county_mapping, distance_threshold=3.0):
        self.osm_downloader = OpenStreetMapDownloader()
        raw_data_filename_1 = f'{self.roads_dir}/roads_raw_data/{geoid}.gpkg'
        raw_data_filename_2 = f'{self.roads_dir}/roads_raw_data/{adj_geoid}.gpkg'
        raw_data_filename = f'{self.roads_dir}/roads_raw_dat/{geoid}_{adj_geoid}.gpkg'

        if not os.path.exists(raw_data_filename_1):
            state_name = self.states[geoid[:2]]
            if geoid in county_mapping:
                county_name = county_mapping[geoid]
            else:
                county_info = self.census.sf1.state_county('NAME', geoid[:2], geoid[2:])
                county_name = county_info[0]['NAME']
            place_name = f'{county_name}, {state_name}, USA'
            logger.info(f'\tDownloading missing county roads: {geoid}')
            msg = self.osm_downloader.download(place_name=place_name, save_filename=raw_data_filename_1)
            edges = gpd.read_file(raw_data_filename_1, layer='edges')
            if msg == 'fail':
                msg = self.osm_downloader.download_polygon(polygon=polygon, save_filename=raw_data_filename_1)

        if not os.path.exists(raw_data_filename_2):
            state_name = self.states[adj_geoid[:2]]
            if geoid in county_mapping:
                county_name = county_mapping[geoid]
            else:
                county_info = self.census.sf1.state_county('NAME', adj_geoid[:2], adj_geoid[2:])
                county_name = county_info[0]['NAME']
            place_name = f'{county_name}, {state_name}, USA'
            logger.info(f'\tDownloading missing county roads {adj_geoid}')
            self.osm_downloader.download(place_name=place_name, save_filename=raw_data_filename_2)
        if nodes_1 is None:
            nodes_1 = gpd.read_file(raw_data_filename_1, layer='nodes')
            nodes_1 = nodes_1.loc[:, ['geometry']]
        if nodes_2 is None:
            nodes_2 = gpd.read_file(raw_data_filename_2, layer='nodes')
            nodes_2 = nodes_2.loc[:, ['geometry']]

        target_nodes = np.array([
                        np.array([point.y, point.x])
                        for point in nodes_2.geometry
                    ])
        del nodes_2
        polygon_points = set()
        # for source_node in source_nodes:
        logger.info('Finding closest nodes')
        for source_node in nodes_1.geometry:
            used_memory = utils.get_occupied_ram()
            if used_memory > 28:
                logger.info(f'Risky memory usage (in for loop)!!!!: {used_memory}')
                return None, 'risky_memory'
            point = np.array([source_node.y, source_node.x])
            distances = utils.haversine_vec(point, target_nodes)
            close_indices = np.where(distances < distance_threshold)
            if len(close_indices[0]):
                close_target_nodes = target_nodes[close_indices]
                # polygon_points.append(source_node)
                polygon_points.add(tuple(list(point)))
                for target_point in close_target_nodes:
                    polygon_points.add(tuple(list(target_point)))
        logger.info('Found closest nodes')
        del target_nodes
        if used_memory > 28:
            logger.info(f'Risky memory usage!!!! (after deleting target_nodes): {used_memory}')
            return None, 'risky_memory'
        
        if len(polygon_points) != 0:
            polygon_points = np.array([[point[0], point[1]] for point in polygon_points])
            polygon_points = self.clockwise_sort(polygon_points)
            polygon = Polygon([point[::-1] for point in polygon_points])
            del polygon_points
            if used_memory > 28:
                logger.info(f'Risky memory usage!!!!(after deleting polygon_points): {used_memory}')
                return None, 'risky_memory'
            return polygon, 'success'
        return None, 'no_polygon'
    
    def sanity_check(self, raw_data_filename, geoid_geometry, populated_percentage=1.0):
        logger.info(f'Sanity checking roads for county: {raw_data_filename}')
        edges = gpd.read_file(raw_data_filename, layer='edges')
        if edges.crs is None:
            edges.crs = self.projection
        else:
            edges.to_crs(self.projection, inplace=True)
        edges['geometry'] = edges.geometry.centroid
        inner_edges = edges[edges.geometry.within(geoid_geometry)]
        pct_inner_edges = len(inner_edges) / len(edges)
        if pct_inner_edges < 0.5:
            return 'fail', pct_inner_edges, 0, 0
        
        nodes = gpd.read_file(raw_data_filename, layer='nodes')
        if nodes.crs is None:
            nodes.crs = self.projection
        else:
            nodes.to_crs(self.projection, inplace=True)
        minx, miny, maxx, maxy = nodes.total_bounds
        
        extreme_points = [
            Point(minx, miny),
            Point(minx, maxy),
            Point(maxx, miny),
            Point(maxx, maxy),
        ]
        # INSERT_YOUR_CODE
        if len(extreme_points) >= 2:
            # Extract latitude and longitude as arrays (reversed order because point[0]=x=lon, point[1]=y=lat)
            coords = np.array([np.array([pt.y, pt.x]) for pt in extreme_points])
            # Calculate all pairwise distances using haversine_vec
            max_dist = 0
            for i in range(len(coords)):
                dists = utils.haversine_vec(coords[i], coords)
                max_dist = max(max_dist, np.max(dists))
        else:
            logger.info(f'No extreme points found for county, {extreme_points}')
            logger.info(f'Num nodes: {len(nodes)}')
            logger.info(f'minx: {minx}, miny: {miny}, maxx: {maxx}, maxy: {maxy}')
            max_dist = 0
        
        # INSERT_YOUR_CODE
        # Get the extreme points (bounding box corners) of the geoid_geometry
        geoid_minx, geoid_miny, geoid_maxx, geoid_maxy = geoid_geometry.bounds
        geoid_extremes = [
            (geoid_miny, geoid_minx),  # (lat, lon) for (minx, miny)
            (geoid_maxy, geoid_minx),  # (lat, lon) for (minx, maxy)
            (geoid_miny, geoid_maxx),  # (lat, lon) for (maxx, miny)
            (geoid_maxy, geoid_maxx),  # (lat, lon) for (maxx, maxy)
        ]
        geoid_coords = np.array(geoid_extremes)
        if len(geoid_coords) >= 2:
            geoid_max_dist = 0
            for i in range(len(geoid_coords)):
                dists = utils.haversine_vec(geoid_coords[i], geoid_coords)
                geoid_max_dist = max(geoid_max_dist, np.max(dists))
        else:
            geoid_max_dist = 0
        max_road_coverage = max_dist / geoid_max_dist

        node_hull = nodes.unary_union.convex_hull
        node_area = node_hull.area
        geometry_area = geoid_geometry.area

        pct_area_covered = node_area / geometry_area
        
        if max_road_coverage < 0.75*populated_percentage and pct_area_covered < 0.75*populated_percentage:
            return 'fail', pct_inner_edges, max_road_coverage, pct_area_covered
        
        return 'success', pct_inner_edges, max_road_coverage, pct_area_covered
            
    def get_road_graph(self, county_mapping, backup_county_mapping, state_mapping, floodmap=None, empty_polygons = set(), empty_polygons_filepath='', populated_percentage=0.4):
        if floodmap is not None:
            openwater = floodmap.loc[floodmap.FLD_ZONE == 'OPEN WATER'].copy()
            openwater.set_index('GEOID', inplace=True)
        else:
            openwater = None
        self.osm_downloader = OpenStreetMapDownloader()
        """
            FIPS, state name mapping copied from
            https://transition.fcc.gov/oet/info/maps/census/fips/fips.txt
        """
        n = len(self.study_area.GEOID.unique())
        for i, (geoid, geoid_geometry) in enumerate(zip(self.study_area.GEOID, self.study_area.geometry)):
            # zip_filename = f'{self.roads_dir}/roads_raw_data/{geoid}.zip'
            raw_data_filename = f'{self.roads_dir}/roads_raw_data/{geoid}.gpkg'
            road_graph_dirpath = f'{self.roads_dir}/Graphs'
            if not os.path.exists(road_graph_dirpath):
                os.makedirs(road_graph_dirpath)
            nodes_dirpath = f'{self.roads_dir}/Nodes'
            if not os.path.exists(nodes_dirpath):
                os.makedirs(nodes_dirpath)
            
            
            road_graph_dist_filename = f'{self.roads_dir}/Graphs/distance_{geoid}.pkl'
            road_graph_time_filename = f'{self.roads_dir}/Graphs/time_{geoid}.pkl'
            # nodes_filename = f'{self.roads_dir}/Nodes/nodes_{geoid}.pkl'
            # edges_filename = f'{self.roads_dir}/Edges/edges_{geoid}.parquet'
            nodes_filename = f'{self.roads_dir}/Nodes/nodes_{geoid}.shp'
            edges_filename = f'{self.roads_dir}/Edges/edges_{geoid}.shp'
            
            # if os.path.exists(road_graph_dist_filename) and os.path.exists(nodes_filename) and os.path.exists(edges_filename):
            #     logger.info(f'Road graphs already exist: {geoid}')
            #     continue
            if os.path.exists(road_graph_dist_filename) and os.path.exists(road_graph_time_filename) and os.path.exists(raw_data_filename):
                logger.info(f'Road graphs and raw roads data already exist: {geoid}')
                continue
            
            if not os.path.exists(raw_data_filename):
                state_name = state_mapping[geoid[:2]]
                sanity_flag = 'fail'
                place_name = f'{county_mapping[geoid].title()} County, {state_name.title()}, USA'
                download_success = self.osm_downloader.download(place_name=place_name, save_filename=raw_data_filename)
                if download_success == 'success':
                    logger.info(f'Downloaded roads using place name: {geoid}, {place_name}')
                    sanity_flag, pct_inner_edges, max_road_coverage, pct_area_covered = self.sanity_check(raw_data_filename, geoid_geometry, populated_percentage)
                    if sanity_flag == 'fail':
                        edges = gpd.read_file(raw_data_filename, layer='edges')
                        if edges.crs is None:
                            edges.crs = self.projection
                        else:
                            edges.to_crs(self.projection, inplace=True)

                        fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
                        temp_df = gpd.GeoDataFrame(pd.DataFrame([geoid_geometry], columns=['geometry']), geometry='geometry', crs=edges.crs)
                        temp_df.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.1)
                        edges.plot(ax=ax, color='red', linewidth=0.2)
                        plt.savefig(f'{self.roads_dir}/bad_roads_{geoid}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
                        plt.close(fig)

                        logger.info(f'Bad roads downloaded using place name: {geoid}, {place_name}: Edges: {round(pct_inner_edges, 2)}, Edge Coverage: {round(max_road_coverage, 2)}, Area: {round(pct_area_covered, 2)}, Populated Percentage: {round(populated_percentage, 2)}')
                        if os.path.exists(raw_data_filename):
                            os.remove(raw_data_filename)
                    else:
                        logger.info(f'Good roads downloaded using place name: {geoid}, {place_name}: Edges: {round(pct_inner_edges, 2)}, Edge Coverage: {round(max_road_coverage, 2)}, Area: {round(pct_area_covered, 2)}, Populated Percentage: {round(populated_percentage, 2)}')


                if sanity_flag == 'fail':
                    time.sleep(30)
                    if geoid in backup_county_mapping:
                        place_name = f'{backup_county_mapping[geoid].title()}, {state_name.title()}, USA'
                    # place_name = f'County of {county_mapping[geoid].title()}, {state_name.title()}, USA'
                    download_success = self.osm_downloader.download(place_name=place_name, save_filename=raw_data_filename)
                    if download_success == 'success':
                        logger.info(f'Downloaded roads using place name default: {geoid}, {place_name}')
                        sanity_flag, pct_inner_edges, max_road_coverage, pct_area_covered = self.sanity_check(raw_data_filename, geoid_geometry, populated_percentage)
                        if sanity_flag == 'fail':
                            logger.info(f'Bad roads downloaded using place name default: {geoid}, {place_name}: Edges: {round(pct_inner_edges, 2)}, Edge Coverage: {round(max_road_coverage, 2)}, Area: {round(pct_area_covered, 2)}')
                            if os.path.exists(raw_data_filename):
                                os.remove(raw_data_filename)
                        else:
                            logger.info(f'Good roads downloaded using place name default: {geoid}, {place_name}: Edges: {round(pct_inner_edges, 2)}, Edge Coverage: {round(max_road_coverage, 2)}, Area: {round(pct_area_covered, 2)}')
                

                if sanity_flag == 'fail':
                    time.sleep(30)
                    download_success = self.osm_downloader.download_polygon(polygon=geoid_geometry, save_filename=raw_data_filename)
                    sanity_flag = 'fail'
                    if download_success == 'success':
                        logger.info(f'Downloaded roads using polygon: {geoid}')
                        sanity_flag, pct_inner_edges, max_road_coverage, pct_area_covered = self.sanity_check(raw_data_filename, geoid_geometry, populated_percentage)
                        if sanity_flag == 'fail':
                            logger.info(f'Bad roads downloaded using polygon: {geoid}: Edges: {round(pct_inner_edges, 2)}, Edge Coverage: {round(max_road_coverage, 2)}, Area: {round(pct_area_covered, 2)}, Populated Percentage: {round(populated_percentage, 2)}')
                            if pct_inner_edges > 0.5:
                                logger.info(f'Not removing the file because these are the only roads for the county')
                            else:
                                # logger.info(f'Removing the file because these are not the only roads for the county')
                                # if os.path.exists(raw_data_filename):
                                #     os.remove(raw_data_filename)
                                logger.info(f'Keeping the file because these are the only roads I can get for the county')
                        else:
                            logger.info(f'Good roads downloaded using polygon: {geoid}: Edges: {round(pct_inner_edges, 2)}, Edge Coverage: {round(max_road_coverage, 2)}, Area: {round(pct_area_covered, 2)}, Populated Percentage: {round(populated_percentage, 2)}')

                if sanity_flag == 'fail':
                    logger.info(f'ROADS MISSING FOR COUNTY: {geoid}')
                    # print(f'ROADS MISSING FOR COUNTY: {geoid}')
                    # exit()
                    empty_polygons.add(geoid)
                
                # if download_success == 'fail':
                #     logger.info(f'DOWNLOAD FAILED FOR COUNTY: {geoid}')
                #     logger.info(f'initiate empty gdf')


        road_graph_dist_unified = nx.MultiGraph()
        road_graph_time_unified = nx.MultiGraph()
        # nodes_unified = {}
        nodes_unified = []
        edges_unified = []
        for i, (geoid, geoid_geometry) in enumerate(zip(self.study_area.GEOID, self.study_area.geometry)):
            logger.info(f'Working on Road graph for county - {geoid}\t...\t{i+1}/{n}')
            raw_data_filename = f'{self.roads_dir}/roads_raw_data/{geoid}.gpkg'
            road_graph_dist_filename = f'{self.roads_dir}/Graphs/distance_{geoid}.pkl'
            road_graph_time_filename = f'{self.roads_dir}/Graphs/time_{geoid}.pkl'
            nodes_filename = f'{self.roads_dir}/Nodes/nodes_{geoid}.shp'
            edges_filename = f'{self.roads_dir}/Edges/edges_{geoid}.shp'
            water_filename = f'{self.roads_dir}/Water/tl_2020_{geoid}_water.shp'
            linearwater_filename = f'{self.roads_dir}/LinearWater/tl_2020_{geoid}_linearwater.shp'
            # nodes_filename = f'{self.roads_dir}/Nodes/nodes_{geoid}.shp'
            # edges_filename = f'{self.roads_dir}/Edges/edges_{geoid}.shp'
            # if not (os.path.exists(nodes_filename) or os.path.exists(edges_filename)):
            #     continue
            if not os.path.exists(nodes_filename) and os.path.exists(raw_data_filename):
                nodes = gpd.read_file(raw_data_filename, layer='nodes')
                if nodes.crs is None:
                    nodes.crs = self.projection
                else:
                    nodes.to_crs(self.projection, inplace=True)
                # nodes[['osmid', 'geometry']].to_parquet(nodes_filename)
                self.write_shapefile_with_lock(nodes[['osmid', 'geometry']], nodes_filename)
            
            if not os.path.exists(edges_filename) and os.path.exists(raw_data_filename):
                edges = gpd.read_file(raw_data_filename, layer='edges')
                if edges.crs is None:
                    edges.crs = self.projection
                else:
                    edges.to_crs(self.projection, inplace=True)
                edges = self.explode_osmid_column(edges)
                if os.path.exists(water_filename):
                    water = gpd.read_file(water_filename)
                    if water.crs is None:
                        water.crs = self.projection
                    else:
                        water.to_crs(self.projection, inplace=True)
                    edges = utils.fast_annotate_edges_strtree(edges, water, 'flooded_area')
                else:
                    water = None
                if os.path.exists(linearwater_filename):
                    linearwater = gpd.read_file(linearwater_filename)
                    linearwater = linearwater.loc[
                        linearwater['FULLNAME']
                                .str.lower()
                                .str.contains('riv|crk', na=False)
                    ]
                    if linearwater.crs is None:
                        linearwater.crs = self.projection
                    else:
                        linearwater.to_crs(self.projection, inplace=True)
                    edges = utils.fast_annotate_edges_strtree(edges, linearwater, 'flooded_linear')
                else:
                    linearwater = None
                    
                if openwater is not None:
                    edges = utils.fast_annotate_edges_strtree(edges, openwater, 'flooded_openwater')
                
                
                edges['annot_bridge'] = edges.apply(
                    lambda row: "yes" if row.get('flooded_area', 0) == 1 or row.get('flooded_linear', 0) == 1 or row.get('flooded_openwater', 0) == 1 else "no", axis=1
                )
                
                if not 'bridge' in edges.columns:
                    edges['bridge'] = 'no'
                
                # if bridge is yes/* then leave as is
                # if bridge is not yes/* and annot_bridge is yes, set bridge to annot_yes
                # otherwise, leave as is
                # def update_bridge(row):
                #     bridge_val = str(row.get('bridge', 'no')).lower()
                #     annot_bridge_val = str(row.get('annot_bridge', 'no')).lower()
                #     # If bridge is 'yes' or '*', leave as is (original value)
                #     if bridge_val == 'yes' or row.get('bridge', '') == '*':
                #         return row['bridge']
                #     # If annot_bridge is 'yes', set to 'annot_yes'
                #     elif annot_bridge_val == 'yes':
                #         return 'annot_yes'
                #     # Otherwise, leave as is
                #     else:
                #         return row['bridge']
                # edges['bridge'] = edges.apply(update_bridge, axis=1)
                bridge_flag = []
                for bridge_val, annot_bridge_val in zip(edges.bridge, edges.annot_bridge):
                    if bridge_val == 'yes' or bridge_val == '*':
                        bridge_flag.append('yes')
                    elif annot_bridge_val == 'yes':
                        bridge_flag.append('annot_yes')
                    else:
                        bridge_flag.append(bridge_val)
                edges['bridge'] = bridge_flag

                fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
                gpd.GeoDataFrame(pd.DataFrame([geoid_geometry], columns=['geometry']), geometry='geometry', crs=edges.crs).plot(ax=ax, color='lightgray', linewidth=0.2)
                if not water is None:
                    water.plot(ax=ax, color='blue', linewidth=0.2, alpha=0.25)
                if not linearwater is None:
                    linearwater.plot(ax=ax, color='cyan', linewidth=0.2, alpha=0.25)
                if not openwater is None:
                    # print(f'openwater: \n{openwater.head()}')
                    if geoid in openwater.index:
                        curr_openwater = openwater.loc[[geoid]]
                    else:
                        curr_openwater = None
                    # print(f'curr_openwater: \n{curr_openwater}')
                    # exit()
                    if curr_openwater is not None and len(curr_openwater) > 0:
                        curr_openwater.plot(ax=ax, color='green', linewidth=0.2, alpha=0.25)
                if edges[(edges.bridge == 'yes') | (edges.bridge == '*')].shape[0] > 0:
                    edges[(edges.bridge == 'yes') | (edges.bridge == '*')].plot(ax=ax, color='red', linewidth=0.2)
                if edges[(edges.bridge == 'annot_yes')].shape[0] > 0:
                    edges[(edges.bridge == 'annot_yes')].plot(ax=ax, color='pink', linewidth=0.2)
                edges[(edges.bridge != 'yes') & (edges.bridge != '*') & (edges.bridge != 'annot_yes')].plot(ax=ax, color='yellow', linewidth=0.2)
                plt.savefig(f'{self.roads_dir}/Edges/edges_{geoid}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
                # INSERT_YOUR_CODE
                import matplotlib.patches as mpatches

                legend_handles = [
                    mpatches.Patch(color='lightgray', label='County boundary'),
                    mpatches.Patch(color='blue', label='Water (area)'),
                    mpatches.Patch(color='cyan', label='Water (linear)'),
                    mpatches.Patch(color='red', label='Bridge (OSM)'),
                    mpatches.Patch(color='pink', label='Bridge (Annotated)')
                ]
                ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.35, 1.05), frameon=True)
                plt.close(fig)


                if not 'length_h' in edges.columns:
                    edges['length_h'] = self.get_road_lengths(edges.geometry)
                

                
                if not 'maxspeed' in edges.columns:
                    edges['maxspeed'] = '0'
                edges['maxspeed'] = edges.maxspeed.replace('', '0')
                edges['maxspeed'] = edges.maxspeed.fillna('0')
                edges['maxspeed'] = self.maxspeed_to_float(edges)
                data = edges[edges.maxspeed != 0].copy()
                data['highway_encoded'] = self.encode_highway(data.highway)
                maxspeeds = self.map_maxspeeds(data)
                edges['maxspeed_e'] = self.estimate_maxspeeds(edges, maxspeeds)
                if not 'length_h' in edges.columns:
                    edges['length_h'] = self.get_road_lengths(edges.geometry) # distance in miles
                edges['time'] = (edges.length_h / edges.maxspeed_e) * 3600 # time in seconds
                # edges[['osmid', 'from', 'to', 'bridge', 'length_h', 'geometry']].to_parquet(edges_filename)
                self.write_shapefile_with_lock(edges[['osmid', 'from', 'to', 'bridge', 'length_h', 'maxspeed_e', 'time', 'geometry']], edges_filename)

            # if not os.path.exists(road_graph_dist_filename) and os.path.exists(nodes_filename) and os.path.exists(edges_filename):
            if (not os.path.exists(road_graph_dist_filename) or not os.path.exists(road_graph_time_filename)) and os.path.exists(nodes_filename) and os.path.exists(edges_filename):
                logger.info(f'Extract and save road graph for county: {geoid}')
                if not os.path.exists(nodes_filename):
                    logger.info(f'Cannot build graph, nodes does not exist for county: {geoid}')
                    continue
                if not os.path.exists(edges_filename):
                    logger.info(f'Cannot build graph, edges does not exist for county: {geoid}')
                    continue
                if not os.path.exists(raw_data_filename):
                    logger.info(f'Cannot build graph, roads raw data does not exist for county: {geoid}')
                    continue

                self.extract_and_dump(
                    nodes_filename,
                    edges_filename,
                    road_graph_dist_filename,
                    road_graph_time_filename
                    )
            else:
                logger.info(f'Use cached road graphs: {geoid}')
                # try:
                #     os.remove(raw_data_filename)
                # except FileNotFoundError:
                #     logger.info(f'\tFile-{geoid} does not exist')
            
            if os.path.exists(road_graph_dist_filename):
                with open(road_graph_dist_filename, 'rb') as pkl:
                    road_graph_dist = pickle.load(pkl)
            else:
                road_graph_dist = None
            if os.path.exists(road_graph_time_filename):
                with open(road_graph_time_filename, 'rb') as pkl:
                    road_graph_time = pickle.load(pkl)
            else:
                road_graph_time = None
            # with open(nodes_filename, 'rb') as pkl:
            #     nodes = pickle.load(pkl)
            # nodes = gpd.read_file(raw_data_filename, layer='nodes')
            # edges = gpd.read_file(raw_data_filename, layer='edges')
            if os.path.exists(nodes_filename):
                nodes = gpd.read_file(nodes_filename)
            else:
                nodes = None
            if os.path.exists(edges_filename):
                edges = gpd.read_file(edges_filename)
            else:
                edges = None
            
            if nodes is not None:
                if nodes.crs is None:
                    nodes.crs = self.projection
                else:
                    nodes.to_crs(self.projection, inplace=True)
            if edges is not None:
                if edges.crs is None:
                    edges.crs = self.projection
                else:
                    edges.to_crs(self.projection, inplace=True)
                
                edges = self.explode_osmid_column(edges)
            # if not 'bridge' in edges.columns:
            #     edges['bridge'] = 'no'
            # if not 'length_h' in edges.columns:
            #     edges['length_h'] = self.get_road_lengths(edges.geometry)
            if edges is not None:
                edges_unified.append(edges[['osmid', 'from', 'to', 'bridge', 'length_h', 'maxspeed_e', 'time', 'geometry']])
            if nodes is not None:
                nodes_unified.append(nodes[['osmid', 'geometry']])
            if road_graph_dist is not None:
                road_graph_dist_unified = nx.compose(road_graph_dist_unified, road_graph_dist)
            if road_graph_time is not None:
                road_graph_time_unified = nx.compose(road_graph_time_unified, road_graph_time)
        # edges_unified = nx.to_pandas_edgelist(road_graph_dist_unified)
        if len(edges_unified) == 0:
            return None, None, None, None, empty_polygons
        edges_unified = pd.concat(edges_unified)
        nodes_unified = pd.concat(nodes_unified)
        # edges_unified['geometry'] = edges_unified['geometry'].apply(wkt.loads)
        edges_unified = gpd.GeoDataFrame(edges_unified, geometry='geometry', crs=self.projection)
        nodes_unified = gpd.GeoDataFrame(nodes_unified, geometry='geometry', crs=self.projection)
        edges_unified['bridge'] = edges_unified.bridge.astype(str).str.strip()
        logger.info(f'Returning the unified distance {len(road_graph_dist_unified)}, time {len(road_graph_time_unified)} graphs and nodes {len(nodes_unified)} with empty polygons')
        return road_graph_dist_unified, road_graph_time_unified, nodes_unified, edges_unified, empty_polygons
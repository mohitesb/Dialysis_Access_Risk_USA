import requests
import zipfile
import os
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import os
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException
import shutil
import osmnx as ox
import pandas as pd
import logging
import warnings
import utils
logger = logging.getLogger(__name__)
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
warnings.filterwarnings("ignore")
from filelock import FileLock
from pathlib import Path
from shapely.ops import unary_union
import zipfile
from dotenv import load_dotenv
load_dotenv()
class CensusBlockBoundariesDownloader:
    def __init__(
            self,
            ftp_link_template = 'https://www2.census.gov/geo/tiger/TIGER2020PL/LAYER/TABBLOCK/2020/tl_2020_{}_tabblock20.zip',
    ):
        self.ftp_link_template = ftp_link_template
    
    def download(self, geoid, save_dirname):
        ftp_link  = self.ftp_link_template.format(geoid)
        # response = requests.get(ftp_link)
        response = utils.get_url_with_retry(ftp_link, timeout=30, max_retries=5, retry_delay=5)
        if not response is None and response.status_code == 200:
            zip_path = os.path.join(save_dirname, 'data.zip')
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(save_dirname)
            os.remove(zip_path)

        


class CensusBlockInfoDownloader:
    def __init__ (
            self, 
            dec_cb_census_link = 'https://api.census.gov/data/2020/dec/pl?get={}&for=block:*&in=state:{}%20county:{}%20tract:*&key={}',
            dec_api_key = os.environ['CENSUS_API_KEY']
    ):
        self.dec_cb_census_link = dec_cb_census_link
        self.dec_api_key = dec_api_key
    
    def download(self, variables, state_geocode, county_geocode, save_filepath):
        url = self.dec_cb_census_link.format(','.join(variables), state_geocode, county_geocode, self.dec_api_key)
        response = utils.get_url_with_retry(url, timeout=30, max_retries=5, retry_delay=5)
        if response is not None and response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data[1:], columns=data[0])
            df.to_csv(save_filepath)
            return df
        else:
            logger.info(
                f"Error in get_url_with_retry for state={state_geocode} county={county_geocode}"
            )
        return None


class OpenStreetMapDownloader:
    def __init__ (self):
        pass
    def write_gpkg_with_lock(self, G, save_filename):
        dst = Path(save_filename)
        lock = FileLock(str(dst) + '.lock')
        with lock:
            if dst.exists() and dst.stat().st_size > 0:
                return 'success'
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            ox.save_graph_geopackage(G, filepath=tmp)
            os.replace(tmp, dst)
            return 'success'


    def download(self, place_name, save_filename):
        try:
            G = ox.graph_from_place(place_name, network_type='drive', truncate_by_edge=True, retain_all=True)      
        except:
            logger.info(f'No nodes found in the county: {place_name}')
            return 'fail'
        
        self.write_gpkg_with_lock(G, save_filename)
        return 'success'
    
    
    def download_polygon(self, polygon, save_filename):
        try:
            G = ox.graph_from_polygon(polygon, network_type='drive', retain_all=True, truncate_by_edge=True)
            self.write_gpkg_with_lock(G, save_filename)
            return 'success'
        except Exception as e:
            logger.info(f'\t\tNo nodes found in this polygon')
            logger.exception(
                f"FAILED {save_filename} | "
                f"type={type(polygon)} valid={getattr(polygon,'is_valid',None)} "
                f"empty={getattr(polygon,'is_empty',None)} bounds={getattr(polygon,'bounds',None)} | "
                f"err={e}"
            )
            return 'fail'
    
    def download_roads_census_gov(self, geoid, zip_filename, save_dirpath):
        url = self.roads_url_template.format(geoid)
        response = utils.get_url_with_retry(url, timeout=10, max_retries=10, retry_delay=5)
        if response is None or response.status_code != 200:
            logger.info(f'Error in downloading the roads data from census.gov - {geoid}')
            return 'fail'

        with open(zip_filename, 'wb') as f:
            f.write(response.content)
        
        logger.info(f'Saving the roads zip file for geoid: {geoid}')

        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(save_dirpath)
        logger.info(f'Unzipped the roads zip file for geoid: {geoid}')
        os.remove(zip_filename)
        logger.info(f'Removed the roads zip file for geoid: {geoid}')
        return 'success'
    
    def download_water_census_gov(self, geoid, zip_filename, save_dirpath):
        url = self.water_url_template.format(geoid)
        response = utils.get_url_with_retry(url, timeout=30, max_retries=5, retry_delay=5)
        if response is None or response.status_code != 200:
            logger.info(f'Error in downloading the water data from census.gov - {geoid}')
            return 'fail'
        with open(save_filename, 'wb') as f:
            f.write(response.content)
        
        logger.info(f'Saving the water zip file for geoid: {geoid}')

        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(save_dirpath)
        logger.info(f'Unzipped the water zip file for geoid: {geoid}')
        os.remove(zip_filename)
        logger.info(f'Removed the water zip file for geoid: {geoid}')
        return 'success'



class FemaFloodMapDownloader:
    def __init__ (
            self,
            county_geoids,
            chromedriver_path = './chromedriver-linux64/chromedriver',
            downloads_folder = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/FEMA_FloodMaps'),
            fema_floodmap_folder = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/FEMA_FloodMaps'),
            wait_time = 30,
            link = 'https://hazards.fema.gov/femaportal/NFHL/searchResult'
    ):
        self.county_geoids = county_geoids
        # self.chromedriver_path = chromedriver_path
        # self.driver = webdriver.Chrome(executable_path=chromedriver_path)
        
        chrome_prefs = {
            'download.default_directory': downloads_folder,
            'download.prompt_for_download': False,
            'download.directory_upgrade': True,
            'safebrowsing.enabled': True,
        }
        options = Options()
        options.add_experimental_option('prefs', chrome_prefs)
        options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--start-maximized')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')

        service = Service(executable_path=chromedriver_path)
        self.driver = webdriver.Chrome(service=service, options=options)


        self.downloads_folder = downloads_folder
        self.fema_floodmap_folder = fema_floodmap_folder
        self.wait_time = wait_time
        self.link = link

    def _extract_and_delete(self, zip_file_path, extract_to_dir):
        if not os.path.exists(extract_to_dir):
            os.mkdir(extract_to_dir)

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_dir)

        for filename in os.listdir(extract_to_dir):
            if not (filename.startswith('S_FLD_HAZ_AR') or filename.startswith('S_BFE') or filename.startswith('S_GEN_STRUCT') or filename.startswith('S_TRNSPORT')):
                file_path = os.path.join(extract_to_dir, filename)
                os.remove(file_path)
                
    def download(self):
        self.driver.get(self.link)
        self.driver.maximize_window()

        downloaded_county = set()
        next_page_button_xpath = '//*[@id="itemTable2_next"]'
        county_geoid_xpath = '/html/body/form/div/div/table/tbody/tr[{}]/td[1]'
        download_button_xpath = '/html/body/form/div/div/table/tbody/tr[{}]/td[6]/center/a/img'
        while True:
            next_page_button =  WebDriverWait(self.driver,self.wait_time).until(
                    EC.presence_of_element_located((By.XPATH, next_page_button_xpath))
                )            
            next_page_button =  WebDriverWait(self.driver, self.wait_time).until(
                    EC.element_to_be_clickable((By.XPATH, next_page_button_xpath))
                )
            for i in range(1, 11):
                try:
                    county_geoid = self.driver.find_element(By.XPATH, county_geoid_xpath.format(i)).text.split('-')[0]
                except NoSuchElementException:
                    self.driver.close()
                    return
                county_geoid = county_geoid[:6]
                if not county_geoid[:5] in self.county_geoids:
                    continue
                downloaded_county.add(county_geoid)
                download_button = WebDriverWait(self.driver, self.wait_time).until(
                    EC.presence_of_element_located((By.XPATH, download_button_xpath.format(i)))
                )
                download_button = WebDriverWait(self.driver, self.wait_time).until(
                    EC.element_to_be_clickable((By.XPATH, download_button_xpath.format(i)))
                )
                download_button.click()
                not_downloaded = True
                sleep(4)
                while not_downloaded:
                    filenames = list(os.listdir(self.downloads_folder))
                    not_downloaded = any([filename.endswith('.part') or filename.endswith('crdownload') for filename in filenames])
                    sleep(2)
                logger.info(f'Downloaded flood map for county - {county_geoid}')
                sleep(2)
                curr_filename = None
                for filename in filenames:
                    if filename.startswith(county_geoid) and filename.endswith('.zip'):
                        curr_filename = os.path.join(self.downloads_folder, filename)
                        break
                if curr_filename is None:
                    logger.info(f'Error working with - {curr_filename}')
                # extract_to_dir = os.path.join(self.downloads_folder, county_geoid)
                extract_to_dir = os.path.join(self.fema_floodmap_folder, county_geoid)
                self._extract_and_delete(curr_filename, extract_to_dir)
                os.remove(curr_filename)
            try:
                next_page_button.click()
            except:
                return        


import os
import utils
import pandas as pd
import geopandas as gpd
import pickle
import logging
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np





geographic_crs = 'EPSG:4326'
external_ssd_path = os.path.join(utils.EXTERNAL_SSD_ROOT, 'Data')
wad_filepath_template = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/baseline_conditions_node_level_2023/{}/ct_wad_all_cms.csv')

counties_filepath = os.path.join(utils.ESRD_ROOT, 'Data/County_Shapefiles/tl_2020_us_county.shp')
counties = gpd.read_file(counties_filepath)
counties['GEOID'] = counties.GEOID.astype(str).str.zfill(5)
counties.crs = geographic_crs

states = gpd.read_file(os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/state_boundaries/tl_2024_us_state.shp'))
states.rename(columns={'NAME': 'ST_NAME'}, inplace=True)
states['STATEFP'] = states.STATEFP.astype(str).str.zfill(2)
counties['STATEFP'] = counties.STATEFP.astype(str).str.zfill(2)
counties = counties.merge(states[['STATEFP', 'STUSPS', 'ST_NAME']], on='STATEFP')
counties = counties.sort_values(by=['GEOID'])
counties.set_index('STUSPS', inplace=True)

svi_df_filepath = os.path.join(utils.ESRD_ROOT, 'Data/uniform_pipeline_v3/SVI_2020_US/SVI2020_US_tract.gdb')
svi_df = gpd.read_file(svi_df_filepath)
svi_df['FIPS'] = svi_df.FIPS.astype(str).str.zfill(11)
svi_df['county_fips'] = svi_df.FIPS.str[:5]


tract_population_df = pd.read_csv(os.path.join(utils.ESRD_ROOT, 'Data/Urban/tract_level_esrd_populations.csv'))
tract_population_df['tract_fips'] = tract_population_df['tract_fips'].astype(str).str.zfill(11)
tract_population_df['rural_esrd_pct'] = tract_population_df['P1_001N_rural_ESRD_HEMO'] * 100 / tract_population_df['P1_001N_ESRD_HEMO']


# svi_df.set_index('county_fips', inplace=True)
results_dirpath = os.path.join(utils.ESRD_ROOT, "Data/uniform_pipeline_v3/WAD_2023")
os.makedirs(results_dirpath, exist_ok=True)

output_summary_dirpath = os.path.join(results_dirpath, 'regression_summaries')
os.makedirs(output_summary_dirpath, exist_ok=True)

esrd_patients_filepath = os.path.join(utils.ESRD_ROOT, 'Data/esrdCountyLevelData/ESRD_Hemo_2023/esrd_patients.pkl')
esrd_patients = pickle.load(open(esrd_patients_filepath, 'rb'))
esrd_rates_filepath = os.path.join(utils.ESRD_ROOT, 'Data/esrdCountyLevelData/ESRD_Hemo_2023/esrd_rates.pkl')
esrd_rates = pickle.load(open(esrd_rates_filepath, 'rb'))

vulnerability_columns = ['EP_MINRTY', 'EP_POV150', 'EP_AGE65', 'EP_DISABL', 'EP_NOVEH', 'EP_MOBILE', 'EP_MUNIT', 'EP_CROWD']
esrd_columns = ['rural_esrd_pct']
# esrd_columns = []
target_column = 'weighted_avg_distance'
progress_bar = tqdm(enumerate(states.STUSPS.unique()), total=len(states.STUSPS.unique()))
all_wad_df = []
all_regression_summaries = []
for i, (state_abb) in progress_bar:
    print(f'State: {state_abb}')
    curr_counties = counties.loc[state_abb]
    wad_df = []
    for county_geoid in curr_counties.GEOID:
        wad_filepath = wad_filepath_template.format(county_geoid)
        if not os.path.exists(wad_filepath):
            logging.info(f'No WAD file found for {county_geoid}')
            continue
        if wad_filepath.endswith('.csv'):
            curr_wad_df = pd.read_csv(wad_filepath)
        elif wad_filepath.endswith('.parquet'):
            curr_wad_df = pd.read_parquet(wad_filepath)
        else:
            print(f'Unknown file format for {wad_filepath}')
            exit()
        if len(curr_wad_df) > 0:
            wad_df.append(curr_wad_df)
    
    if len(wad_df) == 0:
        continue


    wad_df = pd.concat(wad_df)
    wad_df['census_tract_fips'] = wad_df['census_tract_fips'].astype(str).str.zfill(11)
    wad_df = wad_df.merge(svi_df[vulnerability_columns + ['FIPS']], left_on='census_tract_fips', right_on='FIPS')
    wad_df = wad_df.merge(tract_population_df[['tract_fips', 'rural_esrd_pct']], left_on='census_tract_fips', right_on='tract_fips')
    wad_df.drop(columns=['tract_fips'], inplace=True)
    wad_df = wad_df.dropna()
    wad_df = wad_df.loc[wad_df.weighted_avg_distance > 0]
    
    
    if len(wad_df) > 0:
        all_wad_df.append(wad_df)
    X = wad_df[vulnerability_columns + esrd_columns]
    y = wad_df[target_column].values

    
    print(f'Number of rows: {X.shape}, {len(y)}')
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    X_sm = sm.add_constant(X_scaled, has_constant="add")

    glm = sm.GLM(
        y,
        X_sm,
        family=sm.families.Gamma(sm.families.links.Log())
    )

    res = glm.fit(cov_type='HC3')
    
    # print(f'RMSE: {rmse}, R2: {r2}')
    print(res.summary())
    # Save regression summary to a text file
    # While statsmodels' summary() is designed mainly for text output, you can manually construct a DataFrame:
    import numpy as np
    import pandas as pd

    # Get the summary table as arrays
    coef_table = res.summary2().tables[1]  # summary2 gives the params table as DataFrame
    for column in coef_table.columns:
        coef_table[column] = coef_table[column].astype(float).round(3)

    if len(esrd_columns) == 0:
        summary_df_fname = f'{output_summary_dirpath}/{state_abb}_GLM_summary.csv'
    else:
        summary_df_fname = f'{output_summary_dirpath}/{state_abb}_GLM_summary_with_rural_esrd.csv'
    coef_table.to_csv(summary_df_fname)

    # Save regression summary to a text file (human-readable)
    if len(esrd_columns) == 0:
        summary_txt_fname = f'{output_summary_dirpath}/{state_abb}_GLM_summary.txt'
    else:
        summary_txt_fname = f'{output_summary_dirpath}/{state_abb}_GLM_summary_with_rural_esrd.txt'
    with open(summary_txt_fname, 'w') as f:
        f.write(str(res.summary()))
    all_regression_summaries.append({'state_abb': state_abb, 'coef_table': coef_table})

all_wad_df = pd.concat(all_wad_df)
X = all_wad_df[vulnerability_columns + esrd_columns]
y = all_wad_df[target_column].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
X_sm = sm.add_constant(X_scaled, has_constant="add")
glm = sm.GLM(y, X_sm, family=sm.families.Gamma(sm.families.links.Log()))
res = glm.fit(cov_type='HC3')
print('USA')
print(res.summary())
coef_table = res.summary2().tables[1]
for column in coef_table.columns:
    coef_table[column] = coef_table[column].astype(float).round(3)

if len(esrd_columns) == 0:
    summary_df_fname = f'{output_summary_dirpath}/US_GLM_summary.csv'
else:
    summary_df_fname = f'{output_summary_dirpath}/US_GLM_summary_with_rural_esrd.csv'
coef_table.to_csv(summary_df_fname)

if len(esrd_columns) == 0:
    summary_txt_fname = f'{output_summary_dirpath}/US_GLM_summary.txt'
else:
    summary_txt_fname = f'{output_summary_dirpath}/US_GLM_summary_with_rural_esrd.txt'
with open(summary_txt_fname, 'w') as f:
    f.write(str(res.summary()))

all_regression_summaries.append({'state_abb': 'USA', 'coef_table': coef_table})
all_regression_summaries = pd.DataFrame(all_regression_summaries)

print(all_regression_summaries.head())
all_regression_summaries.to_csv(os.path.join(output_summary_dirpath, 'all_regression_summaries.csv'), index=False)
all_regression_summaries.to_csv('all_regression_summaries.csv', index=False)

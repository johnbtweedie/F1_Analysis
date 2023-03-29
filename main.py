import pandas as pd
import numpy as np
from get_fastf1_data import get_practice_data, get_race_data
from models import random_forest_model, linear_reg_model, svr_model

year  = 2022
gp    = "Bahrain"
fps   = ["FP1","FP2","FP3"]

# Controls (Y/N)
perform_refresh           = "N"
train_linear_model        = "Y"
train_random_forest       = "Y"
train_svr_model           = "Y"
perform_predict_position  = "N"

def refresh_data(yn):
       if yn == "Y":
              practice_data, practice_data_std = get_practice_data()  # returns practice results dataframe and a standardized version
              print(practice_data)
              # return practice_data, practice_data_std

              race_data = get_race_data()  # returns race results dataframe and a standardized version
              print(race_data)

              df = pd.concat([practice_data_std, race_data], axis=1, join="inner")
              df['places made'] = df['GridPosition'] - df['Position']
              df['points finish?'] = np.where(df['Points'] >= 0.1, '1', '0')
              df['top six finish?'] = np.where(df['Points'] >= 7.9, '1', '0')
              df['finish?'] = np.where(df['Status'].isin(['Finished', '+1 Lap', '+2 Laps']), '1', '0')

              cols = ['pace_lap', 'num_laps', 'num_stints', 'overall_speed', 'driver', 'TeamName',
                      'FullName', 'Position', 'GridPosition', 'Time', 'Status', 'Points',
                      'places made', 'points finish?', 'race_pace', 'finish?', 'top six finish?',
                      'pace_s1', 'pace_s2', 'pace_s3', 'soft_fastest_lap', 'medium_fastest_lap']

              df = df[cols]
              df_xy = df[['pace_lap', 'num_laps', 'num_stints', 'overall_speed', 'pace_s1', 'pace_s2', 'pace_s3',
                          'soft_fastest_lap', 'medium_fastest_lap', 'GridPosition', 'Position', 'points finish?',
                          'finish?', 'top six finish?', 'places made']]
              print('saving to csv')
              df_xy.to_csv(r"/Users/johntweedie/Dev/Projects/PN22007 - F1 Analysis/df_xy_temp.csv")
              return df_xy#, practice_data, practice_data_std

       else:
           df_xy = pd.read_csv(r"df_xy_2023-02-22.csv", index_col=0)
           return df_xy
def train_random_forest_c(yn, df):
    if yn == "Y":
        return random_forest_model(df)
def train_linear_model_c(yn, df):
    if yn == "Y":
        return linear_reg_model(df)
def train_svr_model_c(yn, df):
    if yn == "Y":
        return svr_model(df)

# df_xy, practice_data, practice_data_std = refresh_data(perform_refresh)
df_xy = refresh_data(perform_refresh)
df_rf, model_rf = train_random_forest_c(train_random_forest, df_xy)
df_lm, model_lm = train_linear_model_c(train_linear_model, df_xy)
df_svr, model_svg = train_svr_model_c(train_svr_model, df_xy)

df = pd.DataFrame()
df['lm_pred'] = df_lm['y_pred']
df['rf_pred'] = df_rf['y_pred']
df['svr_pred'] = df_svr['y_pred']
df['avg'] = (df['rf_pred'] + df['lm_pred'] + df_svr['y_pred']) / 3
df['obs'] = df_lm['y_obs']

print("done")
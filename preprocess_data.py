import pandas as pd

def format_practice_data(df):
    def standardize(df, col):
        # pass dataframe and column label to return standardize column
        return (df[col] - df[col].mean()) / df[col].std()
    df_reg = pd.DataFrame()
    df_reg['pace_lap'] = df['race_pace']
    df_reg['speed_s1'] = df[['speed_s1_FP1','speed_s1_FP2','speed_s1_FP3']].max(axis=1,skipna=True)
    df_reg['speed_s2'] = df[['speed_s2_FP1','speed_s2_FP2','speed_s2_FP3']].max(axis=1,skipna=True)
    df_reg['speed_fl'] = df[['speed_fl_FP1','speed_fl_FP2','speed_fl_FP3']].max(axis=1,skipna=True)
    df_reg['speed_st'] = df[['speed_st_FP1','speed_st_FP2','speed_st_FP3']].max(axis=1,skipna=True)
    df_reg['num_laps'] = df[['num_laps_FP1','num_laps_FP2','num_laps_FP3']].sum(axis=1,skipna=True)
    df_reg['num_stints'] = df[['num_stints_FP1','num_stints_FP2','num_stints_FP3']].sum(axis=1,skipna=True)
    df_reg['pace_s1'] = df[['pace_s1_FP1', 'pace_s1_FP2', 'pace_s1_FP3']].min(axis=1,skipna=False)
    df_reg['pace_s2'] = df[['pace_s2_FP1', 'pace_s2_FP2', 'pace_s2_FP3']].min(axis=1,skipna=False)
    df_reg['pace_s3'] = df[['pace_s3_FP1', 'pace_s3_FP2', 'pace_s3_FP3']].min(axis=1,skipna=False)
    df_reg.dropna(inplace=True)

    for col in df_reg.columns:
        if col == 'key':
            print('Standardizing columns...')
        elif col in ('soft_median_lap', 'soft_fastest_lap', 'medium_median_lap', 'medium_fastest_lap', 'hard_median_lap', 'hard_fastest_lap'):
            print('01\n'+col)
        elif col in ('pace_lap','soft_fastest_lap', 'medium_fastest_lap', 'pace_s1', 'pace_s2', 'pace_s3'):
            print('02\n'+col)
            print(df_reg[col])
            df_reg[col] = standardize(df_reg, col)*(-1)
        else:
            print('03\n'+col)
            df_reg[col] = standardize(df_reg, col)

    df_reg['soft_fastest_lap'] = standardize(df,'SOFT_fastest_lap') * (-1)
    df_reg['medium_fastest_lap'] = standardize(df,'MEDIUM_fastest_lap')  * (-1)
    df_reg['overall_speed'] = ( df_reg['speed_s1'] + df_reg['speed_s2'] + df_reg['speed_fl'] + df_reg['speed_st'] ) / 4


    #df_reg.drop(labels=['speed_s1', 'speed_s2', 'speed_fl', 'speed_st'], axis=1, inplace=True)
    df['key'] = df.index
    df_reg['key'] = df['key']
    df_reg.set_index('key', inplace=True)


    return df_reg

def standardize_time_column(df, column_name):
    print("Filter out rows with 'nan' or 'NaT' values in "+column_name)
    filtered_df = df[column_name][df[column_name].notnull()]# & (df[column_name] != 'NaT')]
    # normalize data
    filtered_df = (1/filtered_df) / (1/filtered_df).abs().max() # relative to speed (1/time) (fastest = 1)

    return filtered_df

def format_practice_tire_data(year, track, fp):
    # load session
    session = ff1.get_session(year, track, fp)
    weekend = session.event
    session.load()

    # get driver list and initialize all drivers dataframe
    drivers = session.results.Abbreviation
    df_all_drivers = pd.DataFrame()

    # iterate through drivers and compile/format practice data and metrics
    for i, driver in enumerate(drivers):
        print("get_track_practice_data\ngrabbing lap data for {}'s {}...".format(driver, fp))
        accurate_laps = session.laps.pick_driver(driver).pick_quicklaps(threshold=1.07)
        # quick_laps = session.laps.pick_driver(driver).pick_quicklaps(threshold=1.03)
        fastest_lap = session.laps.pick_driver(driver).pick_fastest()

        # print("\ngrabbing soft/med/hard tire runs...")
        # create index key: 'VER_2022_FP1', and initialize empty dataframe with it
        idx = "{}_{}_{}".format(driver, year, fp)
        df_driver = pd.DataFrame(index=[idx])
        df_driver['driver'] = driver

        # iterate through each tire compound and compile/format laptime data
        for compound in compounds:
            median_laptime = accurate_laps.loc[accurate_laps['Compound'] == '{}'.format(compound)].median()
            df_driver['{}_median_laptime_{}'.format(compound, fp)] = median_laptime['LapTime']
            fastest_laptime = accurate_laps.loc[accurate_laps['Compound'] == '{}'.format(compound)].min()
            df_driver['{}_fastest_laptime_{}'.format(compound, fp)] = fastest_laptime['LapTime']

        # concat and replace NaT with nan
        df_all_drivers = pd.concat([df_all_drivers, df_driver])
        df_all_drivers = df_all_drivers.mask(df_all_drivers.isna(), np.nan)

    for col in df_all_drivers.columns:
        if col == 'driver':
            print('Standardizing columns...')
        else:
            print(col)
            # could add suffix to preserve times, group by tire compounds later
            df_all_drivers[col] = standardize_time_column(df_all_drivers, col)

    return df_all_drivers
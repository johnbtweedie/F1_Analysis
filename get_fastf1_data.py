import fastf1 as f1
import pandas as pd
import numpy as np
from preprocess_data import format_practice_data, standardize_time_column


def get_practice_data():
    # Cache enable
    f1.Cache.enable_cache('/Users/johntweedie/Dev/Projects/PN22007 - F1 Analysis/cache')

    session_list = pd.read_csv('/Users/johntweedie/Dev/Projects/PN22007 - F1 Analysis/session_list.csv')

    reg_labels = ['pace_lap', 'pace_s1', 'pace_s2', 'pace_s3',
                  'speed_s1', 'speed_s2', 'speed_fl', 'speed_st',
                  'num_laps', 'num_stints', 'Driver']
    reg_sessions = ['FP1', 'FP2', 'FP3']
    resp_sessions = ['Race']
    compounds = ['SOFT', 'MEDIUM']

    df = pd.DataFrame(columns=['key'])
    df_std = pd.DataFrame(columns=['key'])

    for year, gp, fp in zip(session_list['Year'], session_list['GP'], session_list['FP']):
        print("Grabbing data for: ", year, gp, fp)
        df_all_fp = pd.DataFrame(columns=['key']) # initialize dataframe for combined free practice session results

        for fp_session in fp.split(','):  # for each session in the regressors set, run the api
            print('\n\n', fp_session, " loading...")
            session = f1.get_session(year, gp, fp_session)
            session.load(telemetry=False)


            df_session = pd.DataFrame()  # initialize empty master dataframe for the session
            drivers = pd.unique(session.laps['Driver'])  # create a list of drivers in the session

            print("Filtering laptimes...")
            accurate_laps = list()  # list of accurate lap dataframes for each driver
            df_all_drivers_q = pd.DataFrame()
            for i, driver in enumerate(drivers):  # itterate through all the drivers to filter out inaccurate laps
                print(driver)
                drivers_accurate_laps = session.laps.pick_driver(driver).pick_accurate()
                accurate_laps.append(
                    drivers_accurate_laps)  # keep appending the accurate laps for each driver to a list
                accurate_laps[i].drop(columns=['Time', 'PitOutTime', 'PitInTime',
                                               'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
                                               'LapStartTime', 'Team'], inplace=True)  # drop unnecessary columns

                # create dataframe with single entry for driver, specifying index as driver name
                df_driver = pd.DataFrame(index=[driver])
                df_driver['driver'] = df_driver.index

                # get 'pace' based on laps within 3% (1.03) of fastest laptime, speed from 1.05...
                # populate the driver dataframe with regressor terms
                pace_threshold = 1.05
                speed_threshold = 1.04
                df_driver['pace_lap'] = accurate_laps[i].LapTime[
                    ~(accurate_laps[i].LapTime >= (accurate_laps[i].LapTime.min() * pace_threshold))].mean()
                df_driver['pace_s1'] = accurate_laps[i].Sector1Time[
                    ~(accurate_laps[i].Sector1Time >= (accurate_laps[i].Sector1Time.min() * pace_threshold))].mean()
                df_driver['pace_s2'] = accurate_laps[i].Sector2Time[
                    ~(accurate_laps[i].Sector2Time >= (accurate_laps[i].Sector2Time.min() * pace_threshold))].mean()
                df_driver['pace_s3'] = accurate_laps[i].Sector3Time[
                    ~(accurate_laps[i].Sector3Time >= (accurate_laps[i].Sector3Time.min() * pace_threshold))].mean()
                df_driver['speed_s1'] = accurate_laps[i].SpeedI1[
                    ~(accurate_laps[i].SpeedI1 >= (accurate_laps[i].SpeedI1.max() * speed_threshold))].mean()
                df_driver['speed_s2'] = accurate_laps[i].SpeedI2[
                    ~(accurate_laps[i].SpeedI2 >= (accurate_laps[i].SpeedI2.max() * speed_threshold))].mean()
                df_driver['speed_fl'] = accurate_laps[i].SpeedFL[
                    ~(accurate_laps[i].SpeedFL >= (accurate_laps[i].SpeedFL.max() * speed_threshold))].mean()
                df_driver['speed_st'] = accurate_laps[i].SpeedST[
                    ~(accurate_laps[i].SpeedST >= (accurate_laps[i].SpeedST.max() * speed_threshold))].mean()
                df_driver['num_laps'] = accurate_laps[i].LapNumber.max()
                df_driver['num_stints'] = accurate_laps[i].Stint.max()

                #df_driver = df_driver.mask(df_driver.isna(), np.nan)
                # df_driver['pace_s1'] = df_driver['pace_s1'].dt.total_seconds()
                # df_driver['pace_s2'] = df_driver['pace_s2'].dt.total_seconds()
                # df_driver['pace_s3'] = df_driver['pace_s3'].dt.total_seconds()
                for sector in ['s1', 's2', 's3',]:
                    print(df_driver['pace_' + sector])
                    if pd.isnull(df_driver['pace_s1'].iloc[0]) == False:
                        print('ok')
                        df_driver['pace_' + sector] = df_driver['pace_' + sector].dt.total_seconds()
                    else:
                        df_driver['pace_' + sector] = np.nan

                print('mark2')

                # UPDATE SESSION DATAFRAME
                # use a join or merge here and specify session in suffix
                df_driver = df_driver.fillna(np.nan)

                df_session = pd.concat([df_session, df_driver])

                ########################################################################################################################

                # get tire-specific data

                #########################################################################################################################


                print("get_track_practice_data\ngrabbing lap data for {}'s {}...".format(driver, fp_session))
                quick_laps = session.laps.pick_driver(driver).pick_quicklaps(threshold=1.05)
                fastest_lap = session.laps.pick_driver(driver).pick_fastest()

                # create index key: 'VER_2022_FP1', and initialize empty dataframe with it
                idx = "{}_{}_{}".format(driver, year, gp)
                df_driver_q = pd.DataFrame(index=[idx])
                df_driver_q['driver'] = driver

                # iterate through each tire compound and compile/format laptime data
                for compound in compounds:
                    compound_laptimes = quick_laps.loc[quick_laps['Compound'] == '{}'.format(compound)]['LapTime']
                    compound_laptimes = pd.to_timedelta(compound_laptimes, unit='s', errors='coerce').dt.total_seconds()
                    df_driver_q['{}_median_laptime_{}'.format(compound, fp_session)] = compound_laptimes.median()
                    df_driver_q['{}_fastest_laptime_{}'.format(compound, fp_session)] = compound_laptimes.min()

                # concat and replace NaT with nan
                df_all_drivers_q = pd.concat([df_all_drivers_q, df_driver_q])
                df_all_drivers_q = df_all_drivers_q.mask(df_all_drivers_q.isna(), np.nan)

            print("...all drivers complete\n")
            # standardize (as a pct% of fastest) for all drivers in session
            for col in df_all_drivers_q.columns:
                if col == 'driver':
                    print('% relative standardizing tire time columns...')
                else:
                    print(col)
                    # could add suffix to preserve times, group by tire compounds later
                    df_all_drivers_q[col+'_pct'] = standardize_time_column(df_all_drivers_q, col)

            print('...complete')
            print('Appending data...')
            df_session = df_session.add_suffix('_' + fp_session)
            df_session['key'] = drivers + ('_' + str(
                year) + '_' + gp)  # create unique key for each driver's weekend session (e.g. VER_2019_Bahrain)
            df_all_fp = pd.merge(df_all_fp, df_session, how='outer', on='key')

            df_all_drivers_q['key'] = df_all_drivers_q.index
            df_all_fp = pd.merge(df_all_fp, df_all_drivers_q, how='outer', on='key')
            print('...complete')

        ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####
        #     - standardize each tire-specific fastest and median latpime across all 3 pracitce sessions
        #            - will need to make this more robust, currently cannot handle in one of FP1, FP2, or FP3 columns are missing
        ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####    ####

        for compound in compounds:
            # fastest laptimes
            df_all_fp[compound+'_fastest_lap'] = df_all_fp[['{}_fastest_laptime_FP1'.format(compound), '{}_fastest_laptime_FP2'.format(compound) , '{}_fastest_laptime_FP3'.format(compound)]].min(axis=1)
            df_all_fp[compound + '_fastest_lap' + '_pct'] = standardize_time_column(df_all_fp, (compound + '_fastest_lap'))

            # median laptimes
            df_all_fp[compound + '_median_lap'] = df_all_fp[['{}_median_laptime_FP1'.format(compound), '{}_median_laptime_FP2'.format(compound),'{}_median_laptime_FP3'.format(compound)]].min(axis=1)
            df_all_fp[compound + '_median_lap' + '_pct'] = standardize_time_column(df_all_fp, (compound + '_median_lap'))

        df_all_fp.set_index('key', drop=False, inplace=True)

        print('\n\nComputing overall practice performance variables...')
        df_all_fp['pace_lap_FP1'] = pd.to_timedelta(df_all_fp['pace_lap_FP1'], unit='s', errors='coerce').dt.total_seconds()
        df_all_fp['pace_lap_FP2'] = pd.to_timedelta(df_all_fp['pace_lap_FP2'], unit='s', errors='coerce').dt.total_seconds()
        df_all_fp['pace_lap_FP3'] = pd.to_timedelta(df_all_fp['pace_lap_FP3'], unit='s', errors='coerce').dt.total_seconds()
        df_all_fp['race_pace'] = df_all_fp[['pace_lap_FP1', 'pace_lap_FP2', 'pace_lap_FP3']].min(axis=1,skipna=True)  # minimum of filtered avg laptimes for each session
        df_all_fp['speed_s1'] = df_all_fp[['speed_s1_FP1', 'speed_s1_FP2', 'speed_s1_FP3']].max(axis=1, skipna=False)
        df_all_fp['speed_s2'] = df_all_fp[['speed_s2_FP1', 'speed_s2_FP2', 'speed_s2_FP3']].max(axis=1, skipna=False)
        df_all_fp['speed_fl'] = df_all_fp[['speed_fl_FP1', 'speed_fl_FP2', 'speed_fl_FP3']].max(axis=1, skipna=False)
        df_all_fp['speed_st'] = df_all_fp[['speed_st_FP1', 'speed_st_FP2', 'speed_st_FP3']].max(axis=1, skipna=False)
        df_all_fp['speed_ovr'] = df_all_fp['speed_s1'] + df_all_fp['speed_s2'] + df_all_fp['speed_fl'] + df_all_fp['speed_st']
        print('...complete')

        ################################
        # Send to process data
        # Should convert all data to total seconds before this
        ################################

        print('\n\nformatting for models...\n\n')
        df_all_fp_std = df_all_fp.copy()
        df_all_fp_std = format_practice_data(df_all_fp) # format regressors for all 3 practices in a given weekend, standardizes data
        print(df_all_fp_std)

        ################################
        # Append all to master dataframe
        ################################
        df = df.append(df_all_fp)
        df_std = df_std.append(df_all_fp_std)


    df.set_index('key', inplace=True)
    return df, df_std

def get_race_data():
    # Cache enable
    f1.Cache.enable_cache('/Users/johntweedie/Dev/Projects/PN22007 - F1 Analysis/cache')

    session_list = pd.read_csv('/Users/johntweedie/Dev/Projects/PN22007 - F1 Analysis/session_list.csv')

    reg_labels = ['Driver']
    reg_session = 'Race'

    df = pd.DataFrame(columns=['key'])
    df_std = pd.DataFrame(columns=['key'])

    for year, gp in zip(session_list['Year'], session_list['GP']):
        print("Grabbing data for: ", year, gp)

        print(reg_session, " loading...")
        session = f1.get_session(year, gp, reg_session)
        session.load(telemetry=False)

        df_session = pd.DataFrame()  # initialize empty master dataframe for the session
        drivers = pd.unique(session.laps['Driver'])  # create a list of drivers in the session

        print("Filtering laptimes...")
        accurate_laps = list()  # list of accurate lap dataframes for each driver

        results = pd.DataFrame(session.results)
        print(results)
        results.set_index('Abbreviation', inplace=True)

        for i, driver in enumerate(drivers):  # itterate through all the drivers to filter out inaccurate laps
            print(driver)
            drivers_accurate_laps = session.laps.pick_driver(driver).pick_accurate()
            accurate_laps.append(
                drivers_accurate_laps)  # keep appending the accurate laps for each driver to a list
            accurate_laps[i].drop(columns=['Time', 'PitOutTime', 'PitInTime',
                                           'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
                                           'LapStartTime', 'Team'], inplace=True)  # drop unnecessary columns

            # create dataframe with single entry for driver, specifying index as driver name
            df_driver = pd.DataFrame(index=[driver])
            df_driver['driver'] = df_driver.index

            # get 'pace' based on laps within 7% (1.07) of fastest laptime, speed from 1.05...
            # populate the driver dataframe with regressor terms
            pace_threshold = 1.07
            speed_threshold = 1.10
            df_driver['pace_lap'] = accurate_laps[i].LapTime[
                ~(accurate_laps[i].LapTime >= (accurate_laps[i].LapTime.min() * pace_threshold))].mean()
            df_driver['pace_s1'] = accurate_laps[i].Sector1Time[
                ~(accurate_laps[i].Sector1Time >= (accurate_laps[i].Sector1Time.min() * pace_threshold))].mean()
            df_driver['pace_s2'] = accurate_laps[i].Sector2Time[
                ~(accurate_laps[i].Sector2Time >= (accurate_laps[i].Sector2Time.min() * pace_threshold))].mean()
            df_driver['pace_s3'] = accurate_laps[i].Sector3Time[
                ~(accurate_laps[i].Sector3Time >= (accurate_laps[i].Sector3Time.min() * pace_threshold))].mean()
            df_driver['speed_s1'] = accurate_laps[i].SpeedI1[
                ~(accurate_laps[i].SpeedI1 >= (accurate_laps[i].SpeedI1.max() * speed_threshold))].mean()
            df_driver['speed_s2'] = accurate_laps[i].SpeedI2[
                ~(accurate_laps[i].SpeedI2 >= (accurate_laps[i].SpeedI2.max() * speed_threshold))].mean()
            df_driver['speed_fl'] = accurate_laps[i].SpeedFL[
                ~(accurate_laps[i].SpeedFL >= (accurate_laps[i].SpeedFL.max() * speed_threshold))].mean()
            df_driver['speed_st'] = accurate_laps[i].SpeedST[
                ~(accurate_laps[i].SpeedST >= (accurate_laps[i].SpeedST.max() * speed_threshold))].mean()
            df_driver['num_laps'] = accurate_laps[i].LapNumber.max()
            df_driver['num_stints'] = accurate_laps[i].Stint.max()

            # get race results
            rs = ['TeamName', 'FullName', 'Position', 'GridPosition',
                  'Time', 'Status', 'Points']
            for r in rs:
                df_driver[r] = results.loc[driver][r]

            # merge to full session df
            df_session = pd.concat([df_session, df_driver])


        print('...complete')
        df_session['key'] = drivers + ('_' + str(
            year) + '_' + gp)  # create unique key for each driver's weekend session (e.g. VER_2019_Bahrain)
        print('...complete')
        df_session.set_index('key', inplace=True)

        df = df.append(df_session)

    df.rename(columns={"pace_lap": "race_pace", "num_stints": "num_pits"}, inplace=True)
    print(df.columns)
    df.drop(['num_laps'], axis=1, inplace=True)
    return df

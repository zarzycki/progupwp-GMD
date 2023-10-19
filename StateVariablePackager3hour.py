#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.stats import skew
import xarray as xr
import pandas as pd
import datetime
import os
import matplotlib.patheffects as path_effects
import sys

#for loops with multiple ranges
import itertools as it

### https://stackoverflow.com/questions/5423381/checking-if-sys-argvx-is-defined
### This function queries a given index in an arg list to see if it exists
### Originally added so that a configs file can be passed in via arglist, but if not
### code will default to using "configs.csv" in the base directory
def get_arg(index):
    try:
        sys.argv[index]
    except IndexError:
        return ''
    else:
        return sys.argv[index]

#CMZ adding general volume path
VOLNAME=get_arg(1)
RAWDATA=get_arg(2)
thisxStr=get_arg(3)

#xStrs = ['x001', 'x101', 'x201', 'x202', 'x203', 'x204']
if bool(thisxStr):
    xStrs = [thisxStr]
else:
    xStrs = ['x001', 'x101', 'x201', 'x202', 'x203', 'x204']

if not bool(VOLNAME):
    print("Need to specify VOLNAME")
    quit()

# strings which name the missions, model configurations, hour blocks, and variables

MissionNames = ['Atalante_Meteomodem', 'Atalante_Vaisala' , 'BCO_Vaisala'     , \
                'Meteor_Vaisala'     , 'MS-Merian_Vaisala', 'RonBrown_Vaisala']

PlotVarStrings     = ['T',  'Q', 'U', 'V', 'Hwind', 'theta']
OriginalVarStrings = ['ta', 'q', 'u', 'v', 'Hwind', 'theta']
#PlotVarStrings     = ['T',  'Q', 'U', 'V', 'Hwind', 'theta','PMID']
#OriginalVarStrings = ['ta', 'q', 'u', 'v', 'Hwind', 'theta','p']

# THESE FIRST 3 BLOCKS ARE FOR IF THE USER WANTS TO GET ERRORS FOR PARTICULAR HOUR BLOCKS

#  VARIABLES TO BE SET BY THE USER:
HourBlockLength  =  3    # set how many hours you want to be in a time group (must be a factor of 24)
UTCorLocal       = 'UTC'  # do you want to group by UTC time or local time? (put 'UTC' or 'Local Time' as a string)
OverallStartHour =  0     # hour of the day you want to start the grouping brackets
# (whether OverallStartHour is in UTC or local depends on previous variable)
# (ie starting at 2 for 6-hour blocks means grouping 02:00-08:00 and 08:00-14:00 and so on)

# if the user chooses local time, set the start time back 4 hours (Barbados is in UTC-04:00)
if (UTCorLocal == 'UTC'):
    OverallStartHour = OverallStartHour
elif (UTCorLocal == 'Local Time'):
    OverallStartHour = OverallStartHour - 4

# if the UTC start is before midnight, wrap back to the evening hours
if (OverallStartHour < 0):
    OverallStartHour = OverallStartHour + 24


# start hour for file names (00 for starting at 00:00 UTC)
OverallStartHourStr = str(OverallStartHour).zfill(2)


#if a good HourBlockLength was chosen, go ahead and do these calculations
if (np.mod(24, HourBlockLength) == 0):

    NumOfHourBlocks  = np.int( 24/HourBlockLength )


    # CREATE A STRING ARRAY FOR HOUR BLOCK NAMES to tag on to each variable in the format StarthourEndhour
    HourBlockStrings = np.full(NumOfHourBlocks, '    ') # MUST BE 4 SPACES FOR FILLING!!!!
    for blocki in range(0, NumOfHourBlocks):

        # create 2-digit hour strings for the start of a block and the end of a block
        BlockStartHour = int(OverallStartHour + (blocki * HourBlockLength))
        BlockEndHour   = int(BlockStartHour + HourBlockLength)

        # if the hour block goes over midnight, cycle around to the begining of the day
        if (BlockEndHour >= 24):
            BlockEndHour = int(BlockEndHour - 24)

        BlockStartHourStr = str(BlockStartHour).zfill(2)
        BlockEndHourStr = str(BlockEndHour).zfill(2)

        # fill the array in the format StarthourEndhour (0001 is 00:00-01:00)
        HourBlockStrings[blocki] = BlockStartHourStr + BlockEndHourStr


    # fill a 2D array for which hour strings to include in each block
    # example: HourBlockLength of 3 would create a 8x3 array (hourblocks x hours included in that block)
    DefaultHours = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', \
                    '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']

    # rearrange these hours to start at the chosen starting hour
    # (empty array needs to be filled with empty strings of length 2 for propper string fillage)
    StartCycleHours = np.full( 24, '  ')

    for dayhouri in range(0,24):
        if (dayhouri + OverallStartHour < 24):
            StartCycleHours[dayhouri] = DefaultHours[dayhouri + OverallStartHour]
        elif (dayhouri + OverallStartHour >=24): # when you go over, take the hours from the begning of the day
            StartCycleHours[dayhouri] = DefaultHours[dayhouri + OverallStartHour - 24]
        else:
            print('Error Selecting Hours in the day!')

    # reattange into the (hourblocks x hours included in that block) shape
    HourStringsIncluded = np.reshape( StartCycleHours, [ NumOfHourBlocks, HourBlockLength ])

else:
    print('HourBlockLength must be a factor of 24: (1, 2, 3, 4, 6, 8, 12, 24)')


# create a variable for counting the number of soundings in each hour block
#(8 slots in HourBlockStrings order (0003 then 0306 then 0609...))
SoundCounters = np.full( np.size(HourBlockStrings), 0)


# THE POINT OF THESE 2 NESTED LOOPS IS JUST TO COUNT HOW MANY SOUNDINGS ARE IN EACH 3-HOUR CHUNK
for missi in range(0, np.size(MissionNames ) ): # loop over each sounding file
    MISSIONNAME = MissionNames[missi]

    # Download each data file for some given model config and lead time just to count the soudings
    StorageFolder = VOLNAME+'/ThesisData/StephanSoundings/WithCESMdata/x001/'
    dummyStorageFile = 'EUREC4A_' + MISSIONNAME + '-RS_L2_v3.0.0-x001_LeadDay1WithTh.nc'

    dummySoundingDataPlus = xr.open_dataset( StorageFolder + dummyStorageFile, engine='netcdf4')

    # number of altitude levels in the sounding data (should be 3000)
    NumAlts = np.size(dummySoundingDataPlus.alt)
    NumSoundings = np.size(dummySoundingDataPlus.sounding)

    # create a numpy array of the launch times in this sounding file
    LaunchTimeArray = np.array(dummySoundingDataPlus.launch_time)

    for soundi in range(0, NumSoundings): # loop over each sounding

        # the hour string of this particular sounding ie '00' or '01', or '23'...
        SoundHour = str(LaunchTimeArray[soundi])[11:13]

        for hourblocki in range(0, np.size(HourBlockStrings) ):
            HourBlockStr = HourBlockStrings[hourblocki]
            # HourBlockStr is '0003' or '0306' or '0609' or ...

            # if this particular sounding is in this block hour, up the counter by 1
            if any(SoundHour == HourStringsIncluded[hourblocki]):
                SoundCounters[hourblocki] = SoundCounters[hourblocki] + 1


# create model output storage variables of size (Soundings) x (Altitudes) (for both errors and values)
for vari in range(0, np.size(OriginalVarStrings) ): # loop over all variables you want

    OriginalVarStr = OriginalVarStrings[vari] # name of variable in data
    PlotVarStr     = PlotVarStrings[vari]     # name you want in your variables


    for hourblocki in range(0, np.size(HourBlockStrings) ):
        HourBlockStr = HourBlockStrings[hourblocki]
        # HourBlockStr is '0003' or '0306' or '0609' or ...

        # OBSERVATIONS
        # OBSERVATIONS
        exec(PlotVarStr + '_Value' + HourBlockStr + 'Obs = \
             np.full( [SoundCounters[hourblocki], NumAlts], np.nan) ')

        for xstri in range(0, np.size(xStrs)):
            xStr = xStrs[xstri]

            for lead in [1]: #range(0,3):

                exec(PlotVarStr + '_Value' + HourBlockStr + xStr + 'LeadDay' + str(lead) + \
                     ' = np.full( [SoundCounters[hourblocki], NumAlts], np.nan) ')

                exec(PlotVarStr + '_Error' + HourBlockStr + xStr + 'LeadDay' + str(lead) + \
                     ' = np.full( [SoundCounters[hourblocki], NumAlts], np.nan) ')


# THE POINT OF THESE 5 NESTED LOOPS IS TO ACTUALLY STORE 3-HOURLY DATA IN ARRAYS

# OBSERVATIONS
print('observations\n')

# create a variable for counting the number of soundings in each hour block again (set all to 0)
#(8 slots in HourBlockStrings order (0003 then 0306 then 0609...))
blocksoundi = np.full( [np.size(PlotVarStrings), np.size(HourBlockStrings)], 0)

for missi in range(0, np.size(MissionNames) ): # loop over each sounding file
    MISSIONNAME = MissionNames[missi]

    print(MISSIONNAME)

    # Download the observational data in soundings
    StorageFolder = RAWDATA+'/DATA/StephanSoundings/OriginalDownloads/'
    StorageFile = 'EUREC4A_' + MISSIONNAME + '-RS_L2_v3.0.0.nc'

    SoundingData = xr.open_dataset( StorageFolder + StorageFile, engine='netcdf4')

    # create a numpy array of the launch times in this sounding file
    LaunchTimeArray = np.array(SoundingData.launch_time)

    for vari in range(0,np.size(OriginalVarStrings) ): # loop over all variables you want

        OriginalVarStr = OriginalVarStrings[vari] # name of variable in data
        PlotVarStr     = PlotVarStrings[vari]     # name you want in your variables

        # get a variable array from the xarray for code speed
        if (PlotVarStr == 'Q'): # convert Q from kg/kg to g/kg
            exec( 'VarValArray = np.array(SoundingData.' + OriginalVarStr + ') * 1000')

        elif (PlotVarStr == 'Hwind'): # calculation for horizontal wind
            VarValArray = np.array( ((SoundingData.u**2         + SoundingData.v        **2) ** 0.5) )

        else:
            exec( 'VarValArray = np.array(SoundingData.' + OriginalVarStr + ')')
            exec( 'VarErrArray = np.array(SoundingData.' + OriginalVarStr + ')')

        for hourblocki in range(0,np.size(HourBlockStrings) ):
            HourBlockStr = HourBlockStrings[hourblocki]
            # HourBlockStr is '0003' or '0306' or '0609' or ...

            # loop over each sounding in the current mission
            for soundi in range(0, np.size(SoundingData.sounding) ):

                # the hour of the launch for this particular sounding
                SoundHour = str(LaunchTimeArray[soundi])[11:13]

                # if the launch hour is in a particular block, inlcude the data in the storage variables
                if any(SoundHour == HourStringsIncluded[hourblocki]):

                    exec(PlotVarStr + '_Value' + HourBlockStr + 'Obs' + \
                         '[blocksoundi[vari, hourblocki],:] = VarValArray[soundi]')

                    blocksoundi[vari, hourblocki] = blocksoundi[vari, hourblocki] + 1

#MODEL OUTPUT
# MODEL OUTPUT
for xStri in range(0,np.size(xStrs) ): # loop over all model configurations
    xStr = xStrs[xStri]

    print('\n' + xStr)

    for lead in [1]: #range(0,3): # loop over each forecast lead time

        print('\n' + 'Lead Day ' + str(lead) + '\n')

        # create a variable for counting the number of soundings in each hour block again (set all to 0)
        #(8 slots in HourBlockStrings order (0003 then 0306 then 0609...))
        blocksoundi = np.full( [np.size(PlotVarStrings), np.size(HourBlockStrings)], 0)

        for missi in range(0, np.size(MissionNames) ): # loop over each sounding file
            MISSIONNAME = MissionNames[missi]

            print(MISSIONNAME)

            # Download a specific data array for a particular mission, model config, and lead day
            StorageFolder = VOLNAME+'/ThesisData/StephanSoundings/WithCESMdata/' + xStr + '/'
            StorageFile = 'EUREC4A_' + MISSIONNAME + '-RS_L2_v3.0.0-' + xStr + '_LeadDay' + str(lead) + \
            'WithTh.nc'

            SoundingDataPlus = xr.open_dataset( StorageFolder + StorageFile, engine='netcdf4')
#             print(StorageFolder + StorageFile)

            # create a numpy array of the launch times in this sounding file
            LaunchTimeArray = np.array(SoundingDataPlus.launch_time)


            for vari in range(0,np.size(OriginalVarStrings) ): # loop over all variables you want
#                 print('new variable')

                OriginalVarStr = OriginalVarStrings[vari] # name of variable in data
                PlotVarStr     = PlotVarStrings[vari]     # name you want in your variables

                # get a variable array from the xarray for code speed
                if (PlotVarStr == 'Q'): # convert Q from kg/kg to g/kg
                    exec( 'VarValArray = np.array(SoundingDataPlus.' + OriginalVarStr + '_cesmval) * 1000')
                    exec( 'VarErrArray = np.array(SoundingDataPlus.' + OriginalVarStr + '_cesmerr) * 1000')

                elif (PlotVarStr == 'Hwind'): # calculation for horizontal wind
                    VarValArray = np.array( ((SoundingDataPlus.u_cesmval**2 + SoundingDataPlus.v_cesmval**2) ** 0.5) )

                    VarErrArray = np.array( ((SoundingDataPlus.u_cesmval**2 + SoundingDataPlus.v_cesmval**2) ** 0.5) - \
                                         ((SoundingDataPlus.u**2         + SoundingDataPlus.v        **2) ** 0.5) )
                else:
                    exec( 'VarValArray = np.array(SoundingDataPlus.' + OriginalVarStr + '_cesmval)')
                    exec( 'VarErrArray = np.array(SoundingDataPlus.' + OriginalVarStr + '_cesmerr)')

                for hourblocki in range(0,np.size(HourBlockStrings) ):
                    HourBlockStr = HourBlockStrings[hourblocki]
                    # HourBlockStr is '0003' or '0306' or '0609' or ...

                    # loop over each sounding in the current mission
                    for soundi in range(0, np.size(SoundingDataPlus.sounding) ):

                        # the hour of the launch for this particular sounding
                        SoundHour = str(LaunchTimeArray[soundi])[11:13]

                        # if the launch hour is in a particular block, inlcude the data in the storage variables
                        if any(SoundHour == HourStringsIncluded[hourblocki]):

                            exec(PlotVarStr + '_Value' + HourBlockStr + xStr + 'LeadDay' + str(lead) + \
                                 '[blocksoundi[vari, hourblocki],:] = VarValArray[soundi]')

                            exec(PlotVarStr + '_Error' + HourBlockStr + xStr + 'LeadDay' + str(lead) + \
                                 '[blocksoundi[vari, hourblocki],:] = VarErrArray[soundi]')

                            blocksoundi[vari, hourblocki] = blocksoundi[vari, hourblocki] + 1


# SAVING DATA SECTION


# OBSERVATIONS
print('Observations\n')
for Var in PlotVarStrings:

    for hourblocki in range(0,np.size(HourBlockStrings) ):
        HourBlockStr = HourBlockStrings[hourblocki]

        exec("StitchedValues = " + Var + "_Value" + HourBlockStr + "Obs")

        # save the overall observations
        SaveFolder = VOLNAME+'/ThesisVariables/'

        if not os.path.exists(SaveFolder):      # make the folder where you are saving the files
            os.makedirs(SaveFolder)

        ValueSaveFile = Var + 'Value_Obs_' + HourBlockStr + '.npy'
        np.save(SaveFolder+ValueSaveFile, StitchedValues)


        # save mean profiles of the observations
        ValueProfile = np.nanmean(StitchedValues,axis=0)
        ValueSaveFile = Var + 'Value_Obs_' + HourBlockStr + 'MeanProfile.npy'

        np.save(SaveFolder+ValueSaveFile, ValueProfile)



# MODEL OUTPUT
for xStri in range(0,np.size(xStrs) ): # loop over all model configurations
    xStr = xStrs[xStri]
    print('\n' + xStr)

    for lead in [1]: #range(0,3): # loop over each forecast lead time
        print('\n' + 'Lead Day ' + str(lead) + '\n')

        for hourblocki in range(0,np.size(HourBlockStrings) ):
            HourBlockStr = HourBlockStrings[hourblocki]

            for Var in PlotVarStrings:

                exec("StitchedValues = " + Var + "_Value" + HourBlockStr + xStr + "LeadDay" + str(lead))
                exec("StitchedErrors = " + Var + "_Error" + HourBlockStr + xStr + "LeadDay" + str(lead))

                # save the overall errors
                SaveFolder = VOLNAME+'/ThesisVariables/'

                if not os.path.exists(SaveFolder):      # make the folder where you are saving the files
                    os.makedirs(SaveFolder)

                ValueSaveFile = Var + 'Value_' + xStr + 'LeadDay' + str(lead) + '_' + HourBlockStr + '.npy'
                np.save(SaveFolder+ValueSaveFile, StitchedValues)

                ErrorSaveFile = Var + 'Error_' + xStr + 'LeadDay' + str(lead) + '_' + HourBlockStr + '.npy'
                np.save(SaveFolder+ErrorSaveFile, StitchedErrors)


                # save mean profiles of the errors
                ErrorProfile = np.nanmean(StitchedErrors,axis=0)

                ProfileSaveFile = Var + 'Error_' + xStr + 'LeadDay' + str(lead) + '_' + HourBlockStr +  \
                'MeanProfile.npy'
                np.save(SaveFolder+ProfileSaveFile, ErrorProfile)


                # save RMSE profiles
                RMSEProfile = np.sqrt( np.nanmean(np.square(StitchedErrors),axis=0) )

                RMSEProfileSaveFile = Var + 'Error_' + xStr + 'LeadDay' + str(lead) + '_' + HourBlockStr + \
                'RMSEProfile.npy'
                np.save(SaveFolder+RMSEProfileSaveFile, RMSEProfile)


# THIS BLOCK SAVES MASSIVE ARRAYS OF CESM ERRORS AT EVERY 10 M IN EVERY SOUNDING
# ONE ARRAY FOR EVERY VARIABLE, EVERY MODEL CONFIG, AND EVERY LEAD DAY

print("in block saving massive arrays")
for Var in OriginalVarStrings:
    print('\n' + Var + '\n')
    if (Var == 'Hwind'):
        continue
    for xStr in xStrs:
        print(xStr)
        for lead in [1]: #range(0,3):
            print(lead)
            for Mission in MissionNames:

                # download each augmented sounding data file
                FileFolder = VOLNAME+'/ThesisData/StephanSoundings/WithCESMdata/' + xStr + '/'
                SoundingFile = 'EUREC4A_' + Mission + '-RS_L2_v3.0.0-' + xStr + \
                '_LeadDay' + str(lead) + 'WithTh.nc'
                SoundingData = xr.open_dataset(FileFolder + SoundingFile, engine='netcdf4')

                # stitch together soundings from each mission
                exec("Errors = np.array(SoundingData." + Var + "_cesmerr)")
                if (Mission == 'Atalante_Meteomodem'):
                    StitchedErrors = Errors
                else:
                    StitchedErrors = np.concatenate( (StitchedErrors,Errors), axis=0)

            if (Var == 'q'):
                StitchedErrors = StitchedErrors * 1000 # convert from [kg/kg] to [g/kg] for mixing ratio


            # SAVING DATA SECTION

            # save the overall errors
            SaveFolder = VOLNAME+'/ThesisVariables/'

            if not os.path.exists(SaveFolder):      # make the folder where you are saving the files
                os.makedirs(SaveFolder)

            ErrorSaveFile = Var + 'Errors_' + xStr + 'LeadDay' + str(lead) + '.npy'
            np.save(SaveFolder+ErrorSaveFile, StitchedErrors)


            # save mean profiles of the errors
            ErrorProfile = np.nanmean(StitchedErrors,axis=0)

            ProfileSaveFile = Var + 'Errors_' + xStr + 'LeadDay' + str(lead) + 'MeanProfile.npy'
            np.save(SaveFolder+ProfileSaveFile, ErrorProfile)


            # save RMSE profiles
            RMSEProfile = np.sqrt( np.nanmean(np.square(StitchedErrors),axis=0) )

            RMSEProfileSaveFile = Var + 'Errors_' + xStr + 'LeadDay' + str(lead) + 'RMSEProfile.npy'
            np.save(SaveFolder+RMSEProfileSaveFile, RMSEProfile)


# SPECIAL BLOCK FOR HORIZONTAL WIND ERROR CALCULATION

# THIS BLOCK SAVES MASSIVE ARRAYS OF CESM ERRORS AT EVERY 10 M IN EVERY SOUNDING
# ONE ARRAY FOR EVERY VARIABLE, EVERY MODEL CONFIG, AND EVERY LEAD DAY


Var = 'Hwind'
print(Var)

for xStr in xStrs:
    print(xStr)
    for lead in [1]: #range(0,3):
        print(lead)
        for Mission in MissionNames:

            # download each augmented sounding data file
            FileFolder = VOLNAME+'/ThesisData/StephanSoundings/WithCESMdata/' + xStr + '/'
            SoundingFile = 'EUREC4A_' + Mission + '-RS_L2_v3.0.0-' + xStr + \
            '_LeadDay' + str(lead) + 'WithTh.nc'
            SoundingData = xr.open_dataset(FileFolder + SoundingFile, engine='netcdf4')

            # stitch together soundings from each mission
            SoundingU = np.array(SoundingData.u)
            SoundingV = np.array(SoundingData.v)
            ModelU = np.array(SoundingData.u_cesmval)
            ModelV = np.array(SoundingData.v_cesmval)

            if (Mission == 'Atalante_Meteomodem'):
                StitchedSoundingU = SoundingU
                StitchedSoundingV = SoundingV
                StitchedModelU = ModelU
                StitchedModelV = ModelV
            else:
                StitchedSoundingU = np.concatenate( (StitchedSoundingU,SoundingU), axis=0)
                StitchedSoundingV = np.concatenate( (StitchedSoundingV,SoundingV), axis=0)
                StitchedModelU    = np.concatenate( (StitchedModelU,ModelU), axis=0)
                StitchedModelV    = np.concatenate( (StitchedModelV,ModelV), axis=0)

        # SAVING DATA SECTION

        StitchedSoundingHwind = np.sqrt( np.square(StitchedSoundingU) + np.square(StitchedSoundingV) )
        StitchedModelHwind    = np.sqrt( np.square(   StitchedModelU) + np.square(   StitchedModelV) )

        StitchedErrors = StitchedModelHwind - StitchedSoundingHwind



        # save the overall errors
        SaveFolder = VOLNAME+'/ThesisVariables/'

        if not os.path.exists(SaveFolder):      # make the folder where you are saving the files
            os.makedirs(SaveFolder)

        ErrorSaveFile = Var + 'Errors_' + xStr + 'LeadDay' + str(lead) + '.npy'
        np.save(SaveFolder+ErrorSaveFile, StitchedErrors)


        # save mean profiles of the errors
        ErrorProfile = np.nanmean(StitchedErrors,axis=0)

        ProfileSaveFile = Var + 'Errors_' + xStr + 'LeadDay' + str(lead) + 'MeanProfile.npy'
        np.save(SaveFolder+ProfileSaveFile, ErrorProfile)


        # save RMSE profiles
        RMSEProfile = np.sqrt( np.nanmean(np.square(StitchedErrors),axis=0) )

        RMSEProfileSaveFile = Var + 'Errors_' + xStr + 'LeadDay' + str(lead) + 'RMSEProfile.npy'
        np.save(SaveFolder+RMSEProfileSaveFile, RMSEProfile)


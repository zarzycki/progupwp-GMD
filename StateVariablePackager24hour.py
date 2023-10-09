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
thisxStr=get_arg(2)

#xStrs = ['x001', 'x101', 'x201', 'x202', 'x203', 'x204']
if bool(thisxStr):
    xStrs = [thisxStr]
else:
    xStrs = ['x001', 'x101', 'x201', 'x202', 'x203', 'x204']

if not bool(VOLNAME):
    print("Need to specify VOLNAME")
    quit()

# THIS BLOCK CREATES ARRAYS WHICH STORE VARIABLES FROM ALL SOUNDINGS FROM PARTICULAR TIMES OF DAY

# VARIABLES TO BE SET BY THE USER:
HourBlockLength  =  24     # set how many hours you want to be in a time group (must be a factor of 24)
UTCorLocal       = 'UTC'  # do you want to group by UTC time or local time? (put 'UTC' or 'Local Time' as a string)
OverallStartHour =  0     # hour of the day you want to start the grouping brackets
# (whether OverallStartHour is in UTC or local depends on previous variable)
# (ie starting at 2 for 6-hour blocks means grouping 02:00-08:00 and 08:00-14:00 and so on)

# strings which name the missions, model configurations, hour blocks, and variables
# that will be looped over

MissionNames = ['Atalante_Meteomodem', 'Atalante_Vaisala' , 'BCO_Vaisala'     , \
                'Meteor_Vaisala'     , 'MS-Merian_Vaisala', 'RonBrown_Vaisala']

#OriginalVarStrings = ['ta', 'q', 'u', 'v', 'Hwind', 'theta','p']
#ObsVarStrings = ['ta', 'q', 'u', 'v', 'Hwind', 'theta','p']
#PlotVarStrings     = ['T' , 'Q', 'U', 'V', 'Hwind', 'theta','PMID']
OriginalVarStrings = ['ta', 'q', 'u', 'v', 'Hwind', 'theta']
ObsVarStrings = ['ta', 'q', 'u', 'v', 'Hwind', 'theta']
PlotVarStrings     = ['T' , 'Q', 'U', 'V', 'Hwind', 'theta']

# OriginalVarStrings = ['Hwind']
# PlotVarStrings     = ['Hwind']

# if the user chooses local time, set the start time back 4 hours (Barbados is in UTC-04:00)
if (UTCorLocal == 'UTC'):
    OverallStartHour = OverallStartHour
elif (UTCorLocal == 'Local Time'):
    OverallStartHour = OverallStartHour - 4

# if the UTC start is before midnight, wrap back to the evening hours
if (OverallStartHour < 0):
    OverallStartHour = OverallStartHour + 24





# if a good HourBlockLength was chosen, go ahead and do these calculations
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
SoundCounters = np.full( NumOfHourBlocks, 0)


# THE POINT OF THESE 2 NESTED LOOPS IS JUST TO COUNT HOW MANY SOUNDINGS ARE IN EACH 3-HOUR CHUNK
for missi in range(0, np.size(MissionNames ) ): # loop over each sounding file
    MISSIONNAME = MissionNames[missi]

    # Download each data file for some given model config and lead time just to count the soudings
#     StorageFolder = VOLNAME+'/DATA/StephanSoundings/WithMoreCESMdataAndTh/x101/'

    StorageFolder = VOLNAME+'/ThesisData/StephanSoundings/WithCESMdata/x001/'
    dummyStorageFile = 'EUREC4A_' + MISSIONNAME + '-RS_L2_v3.0.0-x001_LeadDay1' + 'WithTh.nc'

    dummySoundingDataPlus = xr.open_dataset( StorageFolder + dummyStorageFile, engine='netcdf4')

    # number of altitude levels in the sounding data (should be 3000)
    NumAlts = np.size(dummySoundingDataPlus.alt)
    NumSoundings = np.size(dummySoundingDataPlus.sounding)

    # create a numpy array of the launch times in this sounding file
    LaunchTimeArray = np.array(dummySoundingDataPlus.launch_time)


    for soundi in range(0, NumSoundings ): # loop over each sounding

        # the hour string of this particular sounding ie '00' or '01', or '23'...
        SoundHour = str(LaunchTimeArray[soundi])[11:13]

        for hourblocki in range(0, NumOfHourBlocks ):
            HourBlockStr = HourBlockStrings[hourblocki]
            # HourBlockStr is '0003' or '0306' or '0609' or ...

            # if this particular sounding is in this block hour, up the counter by 1
            if any(SoundHour == HourStringsIncluded[hourblocki]):
                SoundCounters[hourblocki] = SoundCounters[hourblocki] + 1



# create a [ 10ish config x 3 lead x *soundcounter* soudnings x 3100 altitudes] storage variable
# for each variable and hour block

for vari in range(0, np.size(OriginalVarStrings) ): # loop over all variables you want

    OriginalVarStr = OriginalVarStrings[vari] # name of variable in data
    PlotVarStr     = PlotVarStrings[vari]     # name you want in your variables

    for hourblocki in range(0, NumOfHourBlocks ):
        HourBlockStr = HourBlockStrings[hourblocki]
        # HourBlockStr is '0003' or '0306' or '0609' or ...

        # create model output storage variables of size (xStrs) x (Leads) x (Soundings) x (Altitudes)
        exec('Model' + PlotVarStr + HourBlockStr + \
             ' = np.full( [ np.size(xStrs), 3, SoundCounters[hourblocki], NumAlts], np.nan) ')

        # create observation storage variables of size (Soundings) x (Altitudes)
        exec('Obs' + PlotVarStr + HourBlockStr + \
             ' = np.full( [ SoundCounters[hourblocki], NumAlts], np.nan) ')







# THE POINT OF THESE 5 NESTED LOOPS IS TO ACTUALLY STORE 3-HOURLY DATA IN ARRAYS
for xStri in range(0,1): # np.size(xStrs) ): # loop over all model configurations
    xStr = xStrs[xStri]

    print('\n' + xStr)

    for lead in [1]: #range(0,3): # loop over each forecast lead time

        print('\n' + 'Lead Day ' + str(lead) + '\n')

        # create a variable for counting the number of soundings in each hour block again (set all to 0)
        #(8 slots in HourBlockStrings order (0003 then 0306 then 0609...))
        blocksoundindices = np.full( np.size(HourBlockStrings), 0)

        for missi in range(0, np.size(MissionNames) ): # loop over each sounding file
            MISSIONNAME = MissionNames[missi]

            print(MISSIONNAME)

            # Download a specific data array for a particular mission, model config, and lead day
#             StorageFolder = VOLNAME+'/DATA/StephanSoundings/WithMoreCESMdataAndTh/' + xStr + '/'

            StorageFolder = VOLNAME+'/ThesisData/StephanSoundings/WithCESMdata/' + xStr + '/'

            StorageFile = 'EUREC4A_' + MISSIONNAME + '-RS_L2_v3.0.0-' + xStr + '_LeadDay' + str(lead) + \
            'WithTh.nc'

            SoundingDataPlus = xr.open_dataset( StorageFolder + StorageFile, engine='netcdf4')

            # create a numpy array of the launch times in this sounding file
            LaunchTimeArray = np.array(SoundingDataPlus.launch_time)

            for hourblocki in range(0, NumOfHourBlocks):
                HourBlockStr = HourBlockStrings[hourblocki]

                # HourBlockStr is '0003' or '0306' or '0609' or ...

                for soundi in range(0, np.size(SoundingDataPlus.sounding) ): # loop over each sounding

                    # the hour of the launch for this particular sounding
                    SoundHour = str(LaunchTimeArray[soundi])[11:13]

                    # if the launch hour is in a particular block, inlcude the data in the storage variables
                    if any(SoundHour == HourStringsIncluded[hourblocki]):

                        for vari in range(0, np.size(OriginalVarStrings) ): # loop over all variables you want

                            OriginalVarStr = OriginalVarStrings[vari] # name of variable in data
                            PlotVarStr     = PlotVarStrings[vari]     # name you want in your variables


                            # MULTIPLY Q by 1000 to convert to g/kg (but no other variable)
                            if (PlotVarStr == 'Q'):
                                exec('Model' + PlotVarStr + HourBlockStr + '[xStri, lead, blocksoundindices[hourblocki], :]' + \
                                     ' = SoundingDataPlus.' + OriginalVarStr + '_cesmval[soundi] * 1000')
                            elif (PlotVarStr == 'Hwind'): # calculation for horizontal wind
                                exec('ModelHwind' + HourBlockStr + '[xStri, lead, blocksoundindices[hourblocki], :]' + \
                                     ' = ( (SoundingDataPlus.u_cesmval[soundi]**2) + ' + \
                                     '(SoundingDataPlus.v_cesmval[soundi]**2) ) ** 0.5')
                            else:

                                exec('Model' + PlotVarStr + HourBlockStr + '[xStri, lead, blocksoundindices[hourblocki], :]' + \
                                     ' = SoundingDataPlus.' + OriginalVarStr + '_cesmval[soundi]')

                        blocksoundindices[hourblocki] = blocksoundindices[hourblocki] + 1

            print( str(np.sum(blocksoundindices))  + ' soundings included thus far' + '\n')




print('Observations' + '\n')

# create a variable for counting the number of soundings in each hour block again (set all to 0)
#(8 slots in HourBlockStrings order (0003 then 0306 then 0609...))
blocksoundindices = np.full( NumOfHourBlocks, 0)

# REPEAT THE INNER 3 OF THE PREVIOUS 5 NESTED LOOPS JUST FOR OBSERVATIONAL DATA
for missi in range(0, np.size(MissionNames) ): # loop over each sounding file
    MISSIONNAME = MissionNames[missi]

    print(MISSIONNAME)

    # Download sounding data for just x001 lead day 0 just to take the actual observational data from
#     dummyStorageFolder = VOLNAME+'/DATA/StephanSoundings/WithMoreCESMdataAndTh/x101/'

    StorageFolder = VOLNAME+'/ThesisData/StephanSoundings/WithCESMdata/x001/'

    dummyStorageFile = 'EUREC4A_' + MISSIONNAME + '-RS_L2_v3.0.0-x001_LeadDay1' + 'WithTh.nc'

    dummySoundingDataPlus = xr.open_dataset( StorageFolder + dummyStorageFile, engine='netcdf4')

    # create a numpy array of the launch times in this sounding file
    LaunchTimeArray = np.array(dummySoundingDataPlus.launch_time)

    for hourblocki in range(0, NumOfHourBlocks):
        HourBlockStr = HourBlockStrings[hourblocki]

        # HourBlockStr is '0003' or '0306' or '0609' or ...

        for soundi in range(0, np.size(dummySoundingDataPlus.sounding) ): # loop over each sounding
                                                                          # BUT NOT H WIND! (calculated seperate)

            # the hour of the launch for this particular sounding
            SoundHour = str(LaunchTimeArray[soundi])[11:13]

            # if the launch hour is in a particular block, inlcude the data in the storage variables
            if any(SoundHour == HourStringsIncluded[hourblocki]):

                for vari in range(0, np.size(OriginalVarStrings) ): # loop over all variables you want

                    OriginalVarStr = ObsVarStrings[vari] # name of variable in data
                    PlotVarStr     = PlotVarStrings[vari]     # name you want in your variables


                    # MULTIPLY Q by 1000 to convert to g/kg (but no other variable)
                    if (PlotVarStr == 'Q'):
                        exec('Obs' + PlotVarStr + HourBlockStr + '[ blocksoundindices[hourblocki], :]' + \
                             ' = dummySoundingDataPlus.' + OriginalVarStr + '[soundi] * 1000')
                    elif (PlotVarStr == 'Hwind'): # calculation for horizontal wind
                        exec('ObsHwind' + HourBlockStr + '[blocksoundindices[hourblocki], :]' + \
                             ' = ( (dummySoundingDataPlus.u[soundi]**2) + ' + \
                             '(dummySoundingDataPlus.v[soundi]**2) ) ** 0.5')
#                     elif (PlotVarStr == 'Theta'): # calculation for potential temperature
#                         exec('ObsHwind' + HourBlockStr + '[blocksoundindices[hourblocki], :]' + \
#                              ' = ( (dummySoundingDataPlus.u[soundi]**2) + ' + \
#                              '(dummySoundingDataPlus.v[soundi]**2) ) ** 0.5')
                    else:
                        exec('Obs' + PlotVarStr + HourBlockStr + '[ blocksoundindices[hourblocki], :]' + \
                             ' = dummySoundingDataPlus.' + OriginalVarStr + '[soundi]')


                blocksoundindices[hourblocki] = blocksoundindices[hourblocki] + 1

    print( str(np.sum(blocksoundindices))  + ' soundings included thus far' + '\n')


#EUREC4A_Atalante_Meteomodem-RS_L2_v3.0.0-x001_LeadDay1wTurb.nc

# THIS BLOCK CREATES AN ARRAY OF DIFFERENCES BETWEEN MODEL OUTPUT AND OBSERVATIONS FOR EVERY 10 M

for vari in range(0, np.size(OriginalVarStrings) ): # loop over all variables you want
    OriginalVarStr = OriginalVarStrings[vari] # name of variable in data
    PlotVarStr     = PlotVarStrings[vari]     # name you want in your variables
    print(PlotVarStr)

    blocksoundindices[:] = 0
    for hourblocki in range(0, NumOfHourBlocks):
        HourBlockStr = HourBlockStrings[hourblocki]

        exec('Diffs' + PlotVarStr + HourBlockStr + \
             ' = np.full( [ np.size(xStrs), 3, SoundCounters[hourblocki], NumAlts], np.nan) ')
        print('Hours ' + HourBlockStr[0:2] + ':00 to ' + HourBlockStr[2:4] + ':00')

        # create difference output storage variables of size (xStrs) x (Leads) x (Soundings) x (Altitudes)


        # HourBlockStr is '0003' or '0306' or '0609' or ...

        for soundi in range(0, SoundCounters[hourblocki] ): # loop over each sounding



            for xStri in range(0, np.size(xStrs) ):

                for lead in [1]: #range(0,3):

                    # The DIFFS array is the MODEL OUTPUT MINUS THE OBSERVATIONS (positive means over-prediction)
                    exec('Diffs' + PlotVarStr + HourBlockStr + '[xStri, lead, blocksoundindices[hourblocki], :] = ' + \
   'Model' + PlotVarStr + HourBlockStr + '[xStri, lead, blocksoundindices[hourblocki], :] - ' + \
   'Obs'   + PlotVarStr + HourBlockStr +             '[ blocksoundindices[hourblocki], :]')


            blocksoundindices[hourblocki] = blocksoundindices[hourblocki] + 1

        print( str(np.sum(blocksoundindices))  + ' soundings included thus far' + '\n')


# THIS BLOCK STORES MEANS, MEDIANS, PERCENTILES, AND RMSE OF EACH VARIABLE IN ARRAYS

# which percentiles do you want in addition to the median?
LowPercentile  = 25
HighPercentile = 75

# CREATE EMPTY ARRAYS TO STORE PERCENTILES IN

# loop over each model configuration and lead day and variable for model-based arrays
for xStri in range(0,np.size(xStrs) ):
    xStr = xStrs[xStri]

    for lead in [1]: #range(0,3):
        for vari in range(0, np.size(PlotVarStrings ) ):
            PlotVarStr = PlotVarStrings[vari]  # name you want in your variables

            # mean,low percentiles, medians, high percentile, and RMSE array sizes for model data


            exec('Model' + PlotVarStr + 'Mean'  + xStr + 'LeadDay' + str(lead) +              \
                 ' = np.full( [np.size(HourBlockStrings), NumAlts], np.nan)')

            exec('Model' + PlotVarStr + str(LowPercentile) + xStr + 'LeadDay' + str(lead) +  \
                 ' = np.full( [np.size(HourBlockStrings), NumAlts], np.nan)')

            exec('Model' + PlotVarStr + 'Median'+ xStr + 'LeadDay' + str(lead) +              \
                 ' = np.full( [np.size(HourBlockStrings), NumAlts], np.nan)')

            exec('Model' + PlotVarStr + str(HighPercentile) + xStr + 'LeadDay' + str(lead) + \
                 ' = np.full( [np.size(HourBlockStrings), NumAlts], np.nan)')

            exec('Model' + PlotVarStr + 'Bias'  + xStr + 'LeadDay' + str(lead) + \
                 ' = np.full( [np.size(HourBlockStrings), NumAlts], np.nan)')

            exec('Model' + PlotVarStr + 'RMSE'  + xStr + 'LeadDay' + str(lead) + \
                 ' = np.full( [np.size(HourBlockStrings), NumAlts], np.nan)')

# loop over each variable once for observations
for vari in range(0, np.size(PlotVarStrings) ):
    PlotVarStr     = PlotVarStrings[vari]  # name you want in your variables
    exec('Obs' + PlotVarStr + 'Mean' +              ' = np.full( [np.size(HourBlockStrings), NumAlts], np.nan)')
    exec('Obs' + PlotVarStr + str(LowPercentile) +  ' = np.full( [np.size(HourBlockStrings), NumAlts], np.nan)')
    exec('Obs' + PlotVarStr + 'Median' +            ' = np.full( [np.size(HourBlockStrings), NumAlts], np.nan)')
    exec('Obs' + PlotVarStr + str(HighPercentile) + ' = np.full( [np.size(HourBlockStrings), NumAlts], np.nan)')

print("Calculate all the percentiles")

# CALCULATE ALL OF THE PERCENTILES

# loop over each model configuration and lead day and variable for model-based arrays
for xStri in range(0, np.size(xStrs) ):
    xStr = xStrs[xStri]

    print('\n' + xStr)

    for lead in [1]: #range(0,3):

        print('\n' + 'Lead Day ' + str(lead) )

        for vari in range(0, np.size(PlotVarStrings ) ):
            PlotVarStr = PlotVarStrings[vari]  # name you want in your variables

            print(PlotVarStr)

            for hourblocki in range(0, np.size(HourBlockStrings) ):

                # string at the end of the SoundCounter and variable names
                HourBlockStr = HourBlockStrings[hourblocki]

                for alti in range(0, NumAlts):

                    # means
                    exec('Model'   + PlotVarStr + 'Mean' + xStr + 'LeadDay' + str(lead) + \
                         '[ hourblocki, alti ] = ' + 'np.nanmean(Model' + PlotVarStr + HourBlockStr + \
                         '[xStri, lead, : , alti])')

                    # low percentiles
                    exec('Model'   + PlotVarStr + str(LowPercentile) + xStr + 'LeadDay' + str(lead) + \
                         '[ hourblocki, alti ] = ' + 'np.nanpercentile(Model' + PlotVarStr + HourBlockStr + \
                         '[xStri, lead, : , alti], LowPercentile)')

                    # medians
                    exec('Model'   + PlotVarStr + 'Median' + xStr + 'LeadDay' + str(lead) + \
                         '[ hourblocki, alti ] = ' + 'np.nanpercentile(Model' + PlotVarStr + HourBlockStr + \
                         '[xStri, lead, : , alti], 50)')

                    # high percentiles
                    exec('Model'   + PlotVarStr + str(HighPercentile) + xStr + 'LeadDay' + str(lead) + \
                        '[ hourblocki, alti ] = ' + 'np.nanpercentile(Model' + PlotVarStr + HourBlockStr + \
                        '[xStri, lead, : , alti], HighPercentile)')

                    # absolute error (bias)
                    exec('Model'   + PlotVarStr + 'Bias' + xStr + 'LeadDay' + str(lead) + \
                        '[ hourblocki, alti ] = ' + ' (np.nanmean( Diffs' + PlotVarStr + HourBlockStr + \
                        '[xStri, lead, : , alti] ) ) ')

                    # root mean squared error
                    exec('Model'   + PlotVarStr + 'RMSE' + xStr + 'LeadDay' + str(lead) + \
                        '[ hourblocki, alti ] = ' + ' (np.nanmean( (Diffs' + PlotVarStr + HourBlockStr + \
                        '[xStri, lead, : , alti])**2 ) ) ** 0.5')



print('\n' + 'Observations')

# loop over each variable once for observations
for vari in range(0, np.size(PlotVarStrings ) ):
    PlotVarStr = PlotVarStrings[vari]  # name you want in your variables

    print(PlotVarStr)

    for hourblocki in range(0, np.size(HourBlockStrings) ):

        # string at the end of the SoundCounter and variable names
        HourBlockStr = HourBlockStrings[hourblocki]

        for alti in range(0, NumAlts):

            # means
            exec('Obs'   + PlotVarStr + 'Mean' + '[ hourblocki, alti ] = ' + \
            'np.nanmean(Obs' + PlotVarStr + HourBlockStr + '[ : , alti])')

            # low percentiles
            exec('Obs'   + PlotVarStr + str(LowPercentile) + '[ hourblocki, alti ] = ' + \
            'np.nanpercentile(Obs' + PlotVarStr + HourBlockStr + '[ : , alti], LowPercentile) ')

            # medians
            exec('Obs'   + PlotVarStr + 'Median' + '[ hourblocki, alti ] = ' + \
            'np.nanpercentile(Obs' + PlotVarStr + HourBlockStr + '[ : , alti], 50)')

            # high percentiles
            exec('Obs'   + PlotVarStr + str(HighPercentile) +  '[ hourblocki, alti ] = ' + \
            'np.nanpercentile(Obs' + PlotVarStr + HourBlockStr + '[ : , alti], HighPercentile)')


# THIS BLOCK SAVES ALL OF THE MEDIAN/MEAN/PERCENTILE DATA AS .npy files

StartHourStr = str(OverallStartHour)
StartHourStr = StartHourStr.zfill(2)
# HourFolder = VOLNAME+'/VARIABLES/' + str(HourBlockLength) + 'HourBlocks/' + StartHourStr + 'UTCstart/'


SaveFolder = VOLNAME+'/ThesisVariables/'
if not os.path.exists(SaveFolder):
    os.makedirs(SaveFolder)

# All Model Runs
for xStri in range(0, np.size(xStrs) ):
    xStr = xStrs[xStri]
    print('\n' + xStr)
    for lead in [1]: #range(0,3):
        print('\n' + 'Lead Day ' + str(lead) )
        for vari in range(0, np.size(PlotVarStrings ) ):
            PlotVarStr = PlotVarStrings[vari]  # name you want in your variables
            print(PlotVarStr)



# #             SubFolder = HourFolder + xStr + '/Means/'
#             if not os.path.exists(SubFolder):
#                 os.makedirs(SubFolder)

            SaveFile = 'Model' + PlotVarStr + 'Mean' + xStr + 'LeadDay' + str(lead) + '_' + \
                StartHourStr + 'UTC' + 'start' + str(HourBlockLength) + 'HourBlocks' + '.npy'

            exec( "np.save(SaveFolder + SaveFile, Model" + \
                 PlotVarStr + "Mean" + xStr + "LeadDay" + str(lead) + ")")


#             SubFolder = HourFolder + xStr + '/Biases/'
#             if not os.path.exists(SubFolder):
#                 os.makedirs(SubFolder)

            SaveFile = 'Model' + PlotVarStr + 'Bias' + xStr + 'LeadDay' + str(lead) + '_' + \
                StartHourStr + 'UTC' + 'start' + str(HourBlockLength) + 'HourBlocks' + '.npy'

            exec( "np.save(SaveFolder + SaveFile, Model" + \
                 PlotVarStr + "Bias" + xStr + "LeadDay" + str(lead) + ")")


#             SubFolder = HourFolder + xStr + '/RMSEs/'
#             if not os.path.exists(SubFolder):
#                 os.makedirs(SubFolder)

            SaveFile = 'Model' + PlotVarStr + 'RMSE' + xStr + 'LeadDay' + str(lead) + '_' + \
                StartHourStr + 'UTC' + 'start' + str(HourBlockLength) + 'HourBlocks' + '.npy'

            exec( "np.save(SaveFolder + SaveFile, Model" + \
                 PlotVarStr + "RMSE" + xStr + "LeadDay" + str(lead) + ")")


# OBSERVATIONS
print('\n' + 'Observations')
# loop over each variable once for observations
for vari in range(0, np.size(PlotVarStrings ) ):
    PlotVarStr = PlotVarStrings[vari]  # name you want in your variables
    print(PlotVarStr)


    # BEFORE EACH SAVE, CREATE A FOLDER FOR IT IF IT DOES NOT EXIST
#     SubFolder = HourFolder + 'Observations/Means/'
#     if not os.path.exists(SubFolder):
#         os.makedirs(SubFolder)

    SaveFile = '/Obs' + PlotVarStr + 'Mean_' + StartHourStr + 'UTCstart' + str(HourBlockLength) + 'HourBlocks.npy'

    exec( "np.save(SaveFolder + SaveFile, Obs" + PlotVarStr + "Mean" + ")" )




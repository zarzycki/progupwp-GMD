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

# THIS BLOCK CREATES 3 ARRAYS OF LENGTH 1546 STORING THE CAM INDICES OF DATE/ TIME/ COLUMN FOR EACH SOUDNING


# I know ahead of time that there are 1546 total soundings
NumTotalSoundings = 1546

# create empty arrays to store the CAM dates, time indices, and column indices, corresponding to each sounding
SoundingDateStrs = np.full(NumTotalSoundings, '1901-01-01') # empty date
SoundingTimeInds = np.full(NumTotalSoundings, -999) # -999 is the fill value
SoundingNcolInds = np.full(NumTotalSoundings, -999) # -999 is the fill value

# # set the overall sounding counter to 0 to start with
totalsoundi = 0

# loop over all  ,issions to gather all soundings
MissionNames = ['Atalante_Meteomodem', 'Atalante_Vaisala' , 'BCO_Vaisala'     , \
                'Meteor_Vaisala'     , 'MS-Merian_Vaisala', 'RonBrown_Vaisala']

NumMissions = len(MissionNames)

# empty array to log the number of the first sounding of each mission
MissionStartSoundings = np.full(NumMissions, -999) # -999 is a fill value

for missi in range(0, len(MissionNames)):

    MISSIONNAME = MissionNames[missi]

    # log the number of the first sounding of each mission
    MissionStartSoundings[missi] = totalsoundi

    # Download each data file for some given model config and lead time just to count the soudings
    StorageFolder = VOLNAME+'/ThesisData/StephanSoundings/WithCESMdata/x101/'
    dummyStorageFile = 'EUREC4A_' + MISSIONNAME + '-RS_L2_v3.0.0-x101_LeadDay1WithTh.nc'

    dummySoundingDataPlus = xr.open_dataset( StorageFolder + dummyStorageFile, engine='netcdf4')

    # fill those empty arrays
    for soundi in range(0, np.size(dummySoundingDataPlus.sounding)):
        SoundingDateStrs[totalsoundi] = str(np.array(dummySoundingDataPlus.time_incesm[soundi]))[0:10]
        SoundingTimeInds[totalsoundi] = int(dummySoundingDataPlus.time_ind_incesm[soundi])
        SoundingNcolInds[totalsoundi] = int(dummySoundingDataPlus.ncol_ind_incesm[soundi])
        totalsoundi = totalsoundi + 1


# THIS BLOCK CREATES 1546 x NumLev Arrays for CAM U wind, Z height, and U'W' momentum flux for all soundings

# I know ahead of time that there are 1546 total soundings
NumTotalSoundings = 1546

dummyCAMoutput = xr.open_dataset(RAWDATA+'/DATA/LargeDomainCESMoutput/x001/' + \
                       'LeadDay1/FHIST-ne30-ATOMIC-ERA5-x001.cam.h3.2020-01-06-00000.nc', engine='netcdf4')

NumTimes = np.size(dummyCAMoutput.time)
NumNcols = np.size(dummyCAMoutput.ncol)
NumLevs  = np.size(dummyCAMoutput.lev)
NumILevs = np.size(dummyCAMoutput.ilev)


# create empty arrays for all sounding profiles of wind, height, and momentum flux for each sounding column
Zprofiles     = np.full([NumTotalSoundings,  NumLevs], np.nan)

Uprofiles     = np.full([NumTotalSoundings,  NumLevs], np.nan)
UPWPprofiles  = np.full([NumTotalSoundings, NumILevs], np.nan)




for xstri in range(0, len(xStrs)):
    xStr = xStrs[xstri]
    print(xStr)
    print('')

    # loop over all soundings
    for totalsoundi in range(0,10): # len(SoundingTimeInds)):
        print(totalsoundi)

        YYYYMMDD = SoundingDateStrs[totalsoundi]
        TimeInd = SoundingTimeInds[totalsoundi]
        NcolInd = SoundingNcolInds[totalsoundi]

        # do not include dates for which there is no CAM output
        if ( (YYYYMMDD != '2020-02-27') and (YYYYMMDD != '2020-02-28') and (YYYYMMDD != '2020-02-29') and \
             (YYYYMMDD != '2020-03-01') and (YYYYMMDD != 'yyyy-mm-dd') ):

            CAMoutput = xr.open_dataset(RAWDATA+'/DATA/LargeDomainCESMoutput/' + xStr + '/LeadDay1/' +\
                                        'FHIST-ne30-ATOMIC-ERA5-' + xStr + '.cam.h3.' + YYYYMMDD + '-00000.nc', \
                                        engine='netcdf4')

            # collect the individual profiles of wind, height, and momentum flux for each sounding column
            Zprofiles[totalsoundi,:]    = np.array(  CAMoutput.Z3[TimeInd,:,NcolInd])

            Uprofiles[totalsoundi,:]    = np.array(   CAMoutput.U[TimeInd,:,NcolInd])
            UPWPprofiles[totalsoundi,:] = np.array(CAMoutput.upwp[TimeInd,:,NcolInd])


    # calculate effective eddy diffusivity profiles for all sounding columns
    KeffProfiles = np.full([NumTotalSoundings, NumILevs], np.nan)
    dUdZprofiles = np.full([NumTotalSoundings, NumILevs], np.nan)

    for totalsoundi in range(0, NumTotalSoundings):
        for levi in range(0,NumLevs-1):

            dUdZ = (Uprofiles[totalsoundi, levi] - Uprofiles[totalsoundi, levi+1]) / \
                   (Zprofiles[totalsoundi, levi] - Zprofiles[totalsoundi, levi+1])

            dUdZprofiles[totalsoundi, levi+1] = dUdZ
            KeffProfiles[totalsoundi, levi+1] = -1 * (UPWPprofiles[totalsoundi, levi+1] / dUdZ)

    # save these profiles
    SaveFolder = VOLNAME+'/ThesisVariables/'
    SaveFile = 'METEO600dUdZprofiles' + xStr + '.npy'
    if not os.path.exists(SaveFolder):
        os.makedirs(SaveFolder)

    np.save(SaveFolder + SaveFile, dUdZprofiles)


    SaveFolder = VOLNAME+'/ThesisVariables/'
    SaveFile = 'METEO600KeffProfiles' + xStr + '.npy'
    if not os.path.exists(SaveFolder):
        os.makedirs(SaveFolder)

    np.save(SaveFolder + SaveFile, KeffProfiles)




# THIS BLOCK COLLECTS MOMENTUM BUDGET DATA FOR ALL H4 DATA DAYS

# empty array for values of momentum components at each altitude in each sounding
# I know in advance that there is a total of 1546 soundings in all missions
# I know in advance that the soudnings store 3000 values of altitude
NumTotalSoundings = 1546
NumAlts = 3100 # I know ahead of time that there are exactly 3100 altidues in each sounding
MassVarArray = np.full([NumTotalSoundings,NumAlts], np.nan)

# only working with 1 day leads here
lead = 1

# specific turbulence component variables
Vars = ['upwp_bt' , 'upwp_ma' , 'upwp_ta' , 'upwp_tp' , 'upwp_ac' , 'upwp_bp', \
        'upwp_pr1', 'upwp_pr2', 'upwp_pr3', 'upwp_pr4', 'upwp_dp1', 'upwp_cl', 'upwp_mfl', \
        'vpwp_bt' , 'vpwp_ma' , 'vpwp_ta' , 'vpwp_tp' , 'vpwp_ac' , 'vpwp_bp', \
        'vpwp_pr1', 'vpwp_pr2', 'vpwp_pr3', 'vpwp_pr4', 'vpwp_dp1', 'vpwp_cl', 'vpwp_mfl']

# names of the different ships that the soundings come from in different files
MissionNames = ['Atalante_Meteomodem', 'Atalante_Vaisala' , 'BCO_Vaisala'     , \
                'Meteor_Vaisala'     , 'MS-Merian_Vaisala', 'RonBrown_Vaisala']

# CAM configs beings reviewed here
#xStrs = ['x101'] #, 'x201', 'x202', 'x203', 'x204']
# CMZ, let's just keep xStrs from before

# grab an example .h4 file to collect sizes from
dummyH4data = xr.open_dataset(RAWDATA+'/DATA/LargeDomainCESMoutput/x101/h4/LeadDay1/' + \
                               'FHIST-ne30-ATOMIC-ERA5-x101.cam.h4.2020-01-06-00000.nc', engine='netcdf4')

NumDays = 52 # there are 52 days for which H4 data are downloaded
NumH4Times = np.size(dummyH4data.time)
NumH4Levs = np.size(dummyH4data.ilev)
NumH3Levs = np.size(dummyH4data.lev)
NumH4Ncols = np.size(dummyH4data.ncol)

for xstri in range(0, len(xStrs)):
    xStr = xStrs[xstri]
    print(xStr)


    # empty array to fill with the variable values from h4 and h3 data (once per xStr)
    for Var in Vars:
        exec( Var + "AtHandInH4 = np.full( [NumDays, NumH4Times, NumH4Levs, NumH4Ncols], np.nan)" )

    GrandAltsH3 = np.full( [NumDays, NumH4Times, NumH3Levs, NumH4Ncols], np.nan)

#     GET MASSIVE ARRAYS FOR THIS VARIABLE FOR THIS XSTR FOR ALL DAYS
#     loop over all days around the campaign model data were downloaded for
    DayOfData = 0
    for monthi in range(1,3):

        # create 2-digit month number string
        if (monthi < 10):
            MM = '0' + str(monthi)
        elif (monthi >= 10):
            MM = str(monthi)
        else:
            print('Error in creating MM string')


        # loop over 5-31 January and 1-25 February (shifted by lead day)
        if (monthi == 1):
            MonthStartDay = 5 + lead
            MonthEndDay   = 31
        elif (monthi == 2):
            MonthStartDay = 1
            MonthEndDay   = 25 + lead
        else:
            print('Error in choosing month')


        # starting day index for each month
        if (monthi == 1): dayi = 0
        if (monthi == 2): dayi = 27 - lead

        for monthdayi in range(MonthStartDay, MonthEndDay + 1):

            # create 2-digit day number string
            if (monthdayi < 10):
                DD = '0' + str(monthdayi)
            elif (monthdayi >= 10):
                DD = str(monthdayi)
            else:
                print('Error in creating DD string')

            print('Working on 2020-' + MM + '-' + DD)

            # get the H4 data for this day
            H4data = xr.open_dataset(RAWDATA+'/DATA/LargeDomainCESMoutput/' + xStr + '/' + \
                           'h4/LeadDay' + str(lead) + '/FHIST-ne30-ATOMIC-ERA5-' + xStr + \
                           '.cam.h4.2020-' + MM + '-' + DD +'-00000.nc', engine='netcdf4')

            for Var in Vars:
                exec(Var + "AtHandInH4[DayOfData,:,:,:] = np.array(H4data." + Var + ")" )
                #exec("print("+Var + "AtHandInH4)")

            H4data.close()


            # get the H3 data for this day
            H3data = xr.open_dataset(RAWDATA+'/DATA/LargeDomainCESMoutput/' + xStr + '/' + \
                'LeadDay' + str(lead) + '/FHIST-ne30-ATOMIC-ERA5-' + xStr + \
                '.cam.h3.2020-' + MM + '-' + DD +'-00000.nc',\
                engine='netcdf4')

            # altitudes from the H3 data
            GrandAltsH3[DayOfData,:,:,:] = np.array(H3data.Z3)

            H3data.close()


            DayOfData = DayOfData + 1

    # save the massive arrays
    for Var in Vars:

        SaveFolder = VOLNAME+'/ThesisVariables/'
        SaveFile = 'H4Grand_' + Var + xStr + 'Valt.npy'

        if not os.path.exists(SaveFolder):
            os.makedirs(SaveFolder)


        exec("np.save(SaveFolder + SaveFile, " + Var + "AtHandInH4)" )

    # Save the alts
    np.save(VOLNAME+'/ThesisVariables/H4Grand_' + xStr + 'Valt.npy', + \
        GrandAltsH3)

    print('saved')


# THIS BLOCK COLLECTS MOMENTUM BUDGET DATA THAT CORRESPONDS TO SOUNDINGS

for xstri in range(0,len(xStrs)):
    xStr = xStrs[xstri]
    print(xStr)

    SaveFolder = VOLNAME+'/ThesisVariables/'
    SaveFile = 'H4Grand_' + xStr + 'Valt.npy'

    GrandAltsH3 = np.load(SaveFolder + SaveFile)

    for Var in Vars:
        print(Var)

        SaveFolder = VOLNAME+'/ThesisVariables/'
        SaveFile = 'H4Grand_' + Var + xStr + 'Valt.npy'

        VarAtHandInH4 = np.load(SaveFolder + SaveFile)

        # loop over each mission with soundings (keep track of overall sounding count)
        overallsoundi = 0
        for missi in range(0, len(MissionNames) ):

            MISSIONNAME = MissionNames[missi]
            print(MISSIONNAME)

            # Download each data file for some given model config and lead time just to count the soudings
            StorageFolder = VOLNAME+'/ThesisData/StephanSoundings/WithCESMdata/' + xStr + '/'
            StorageFile = 'EUREC4A_' + MISSIONNAME + '-RS_L2_v3.0.0-' + xStr + '_LeadDay' + str(lead) + 'WithTh.nc'

            SoundingData = xr.open_dataset( StorageFolder + StorageFile, engine='netcdf4')

            # get the number of soundings
            NumSoundings = np.size(SoundingData.sounding)

            # get the indices in H4 data of each sounding an altitude
            CAMtimeIs = np.array(SoundingData.time_ind_incesm)
            CAMncolIs = np.array(SoundingData.ncol_ind_incesm)
            CAMaltIs  = np.array(SoundingData.alt_ind_incesm)
            # and what the altitude is and is in CAM
            CAMalts = np.array(SoundingData.alt_incesm)
            SOUNDalts = np.array(SoundingData.alt)

            SoundingLaunchTimes = np.array(SoundingData.time_incesm)


            # loop over number of soundings in this specific mission
            for soundi in range(0, NumSoundings):
                if (np.mod(soundi,100) == 0):
                    print(soundi)

                # retrieve what month and day this sounding is for at 1 km

                MM = str(SoundingLaunchTimes[soundi])[5:7]
                DD = str(SoundingLaunchTimes[soundi])[8:10]

                # convert month and day strings to which day of data it is from
                if ( MM.isdigit() ) and ( MM.isdigit() ):
                    month = int(MM)
                    dayofmonth = int(DD)
                    DayOfData = ( (month - 1) * 31) + dayofmonth - 6
                else: # put in error codes if there is no month and day
                    month = 9999
                    dayofmonth = 9999
                    DayOfData = 9999



                # exclude all soundings taken after H4 data ends (after 26 Feb 2020)
                if (DayOfData<52):

                    # momentum budget variable from the H4 data and altitudes from the H3 data
                    VariableAtHand = VarAtHandInH4[DayOfData,:,:,:]
                    AltsHere = GrandAltsH3[DayOfData,:,:,:]


                    # get the indices in H4 data of each sounding an altitude
                    timei = CAMtimeIs[soundi]
                    ncoli = CAMncolIs[soundi]

                    # lineraly interpolate for each altitude
                    for alti in range(0,NumAlts):

                        # find the corresponding altitude index and how off the altitude is
                        levi    = CAMaltIs[soundi, alti]

                        CAMaltHere = CAMalts[soundi, alti]
                        SOUNDaltHere = SOUNDalts[alti]

                        # if the sounding altitude is above the altitude in CAM, then that is the lower nearest altitude
                        if (SOUNDaltHere > CAMaltHere):
#                             print('above')
                            BOTlevi = levi
                            TOPlevi = levi - 1 # (top is 1 pressure level lesser)
                        else: # otherwise the nearest altitude, is the nearest upper altitude
#                             print('below')
                            BOTlevi = levi + 1 # (bottom is 1 pressure level greater)
                            TOPlevi = levi

                        # if the nearest level is the highest or lowest model levels, set that extremum to be both the
                        # model level 'above' and 'below' the sounding altitude
                        if (levi > NumH4Levs-3):
                            BOTlevi = levi - 3
                            TOPlevi = levi - 4
                        elif (levi == 0):
                            BOTlevi = levi + 1
                            TOPlevi = levi

                        # find what the corresponding altitudes are in the model data at the levels
                        # just above and just below the point in the sounding

                        TopAlt = AltsHere[timei,TOPlevi,ncoli]
                        BotAlt = AltsHere[timei,BOTlevi,ncoli]

                        # how far up from the bottom level to the top level is the soudning altitude
                        if (SOUNDaltHere > CAMaltHere):
                            upcentage = (SOUNDaltHere - CAMaltHere) / (TopAlt-BotAlt)
                        else:
                            upcentage = 1 - ( -1 * ( (SOUNDaltHere - CAMaltHere) / (TopAlt-BotAlt)) )

                        # just a filler because you run into a difference of 0 problem
                        # below the lowest model level
                        if (levi == 0):
                            upcentage = 0


                        MassVarArray[overallsoundi,alti] = \
                                                    (VariableAtHand[timei,BOTlevi,ncoli] * (1-upcentage) ) + \
                                                    (VariableAtHand[timei,TOPlevi,ncoli] *    upcentage)

                # always add 1 to the total sounding count at the end!
                overallsoundi = overallsoundi + 1

        SaveFolder = VOLNAME+'/ThesisVariables/'
        SaveFile = 'SoundingGrand_' + Var + xStr + 'Valt.npy'

        if not os.path.exists(SaveFolder):
            os.makedirs(SaveFolder)

        np.save(SaveFolder + SaveFile, MassVarArray)

        del VariableAtHand








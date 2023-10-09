#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
import pandas as pd
import datetime
import os
import sys

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

if not bool(VOLNAME):
    print("Need to specify VOLNAME")
    quit()

# THIS BLOCK ADDS THETA TO H3 DATA

print("--> Adding theta to h3 data")

# first grab hyam and hybm from .h4 data to later calculate pressure levels

# grab an example .h4 file (all hyam and hymb data are the same across time, column, model config)
dummyH4data = xr.open_dataset(VOLNAME+'/DATA/LargeDomainCESMoutput/x101/h4/LeadDay1/' + \
                               'FHIST-ne30-ATOMIC-ERA5-x101.cam.h4.2020-01-06-00000.nc', engine='netcdf4')
# collect the components of calculating pressure and turn them into numpy arrays
hyamList     = np.array(dummyH4data.hyam)
hybmList     = np.array(dummyH4data.hybm)


# list of model configurations
#xStrs = ['x101'] #, 'x101', 'x201', 'x202', 'x203', 'x204']
if bool(thisxStr):
    xStrs = [thisxStr]
else:
    xStrs = ['x001', 'x101', 'x201', 'x202', 'x203', 'x204']


# loop over all model runs, lead times, and days
for xStr in xStrs:
    print(xStr)
    for lead in range(0,3):
        print('lead day ' + str(lead))
        for DOYi in range(5,31+25+1):

            TimeA = datetime.datetime.now()
            # actual day of year
            DOY = DOYi + lead

            if (DOY <= 31):
                MM = '01'
                DD = str(DOY).zfill(2)
            elif (DOY > 31) and (DOY <= 31 + 29):
                MM = '02'
                DD = str(DOY-31).zfill(2)
            else:
                print('Day of Year out of range')

            MMdDD = MM + '-' + DD
            print(MMdDD)

            TimeB = datetime.datetime.now()
            # path names for downloading files
            H3OGdownloadFolder = VOLNAME+'/DATA/LargeDomainCESMoutput/' + xStr + '/LeadDay' + str(lead) + '/'
            H3OriginalFileName       = 'FHIST-ne30-ATOMIC-ERA5-' + xStr + '.cam.h3.2020-' + MMdDD + '-00000.nc'
            H3OriginalFileNameSansNC = 'FHIST-ne30-ATOMIC-ERA5-' + xStr + '.cam.h3.2020-' + MMdDD + '-00000'
            # actually import the original CESM output data
            H3data = xr.open_dataset(H3OGdownloadFolder + H3OriginalFileName, engine='netcdf4')

            TimeC = datetime.datetime.now()
            # collect temperature and pressure and turn them into numpy arrays
            Temps = np.array(H3data.T)
            SurfPs = np.array(H3data.PS[0,:])

            TimeD = datetime.datetime.now()
            # retrieve the number of times, pressure levels, and columns
            NumTimes = np.size(H3data.time)
            NumLevs  = np.size(H3data.lev)
            NumCols  = np.size(H3data.ncol)

            TimeE = datetime.datetime.now()
            # create an empty array for pressures and calculate them
            Pressures = np.full([NumLevs, NumCols], np.nan)
            for levi in range(0, NumLevs):
                for coli in range(0, NumCols):
                    Pressures[levi, coli] = (hyamList[levi]*100000) + hybmList[levi]*SurfPs[coli]

            TimeF = datetime.datetime.now()
            # create an empty array for potential temperatures and calculate them
            Thetas = np.full(np.shape(Temps), np.nan)
            for timei in range(0,NumTimes):
                for levi in range(0, NumLevs):
                    for coli in range(0, NumCols):
                        Thetas[timei, levi, coli] = Temps[timei, levi, coli] * (100000/(Pressures[levi, coli]))**(0.286)

            TimeG = datetime.datetime.now()
            # ADD XARRAY VARIABLES TO DATA ARRAY FOR CESM VALUES AND CORRESPONDING ERRORS
            ThetaXR = xr.DataArray(Thetas,dims=['time','lev','ncol'],coords=\
                        {'time':H3data.time.values,'lev':H3data.lev.values,'ncol':H3data.ncol.values}, name='THETAC')
            ThetaXR.attrs = H3data.THETAL.attrs
            ThetaXR.attrs['standard_name'] = 'potential_temperature_calculated'
            ThetaXR.attrs['long_name'] = 'potential temperature calculated'

            # add this variable to the H3 data xarray
            NewH3data = xr.merge([H3data, ThetaXR])

            TimeH = datetime.datetime.now()
            # SAVE THE AUGMENTED ARRAY AS A NETCDF FILE
            # make the directory to save it in if it doesn't already exist
            SaveFolder = VOLNAME+'/ThesisData/LargeDomainCESMoutputWithTh/' + xStr + '/LeadDay' + str(lead) + '/'
            if not os.path.exists(SaveFolder):
                os.makedirs(SaveFolder)

            TimeI = datetime.datetime.now()
            NewH3data.to_netcdf(SaveFolder + H3OriginalFileNameSansNC + 'WithTh.nc')

            TimeJ = datetime.datetime.now()


#             print('Time AB = ' + str(TimeB-TimeA) )
#             print('Time BC = ' + str(TimeC-TimeB) )
#             print('Time CD = ' + str(TimeD-TimeC) )
#             print('Time DE = ' + str(TimeE-TimeD) )
#             print('Time EF = ' + str(TimeF-TimeE) )
#             print('Time FG = ' + str(TimeG-TimeF) )
#             print('Time GH = ' + str(TimeH-TimeG) )
#             print('Time HI = ' + str(TimeI-TimeH) )
#             print('Time IJ = ' + str(TimeJ-TimeI) )
#             print('Total Time = ' + str(TimeJ-TimeA) )
#             print('')



# THIS FUNCTION FINDS THE GREAT CIRCLE DISTANCE BETWEEN 2 POINTS

def GreatCircle(lat1, lon1, lat2, lon2):

    # radius of the Earth [m]
    R = 6378137

    # convert lat and lon to radians
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    lon1r = np.radians(lon1)
    lon2r = np.radians(lon2)

    # lat and lon differences
    latdif = lat1r - lat2r
    londif = lon1r - lon2r

    # calculation components
    Apart = np.sin(latdif / 2) ** 2
    Bpart = np.sin(londif / 2) ** 2

    # calculation
    distance = 2 * R * \
        np.arcsin( (Apart + (np.cos(lat1r) * np.cos(lat2r) * Bpart) )**0.5 )

    return distance


# THIS FUNCTION ADDS CESM OUTPUT COORDINATES AND MODEL PREDICTIONS AND ERRORS TO SOUNDING XARRAYS
# LINEAR INTERPOLATION IS USED

               # (Xarray,     string,  int,        array,    array)
def AddCESMcoords(SoundingData, xStr, lead, SoundingVars, CESMvars):

    StartTime = datetime.datetime.now()

    NumOfSoundings = np.size(SoundingData.sounding)

    # The first and last soundings to loop over
    StartSounding = 0
    EndSounding = NumOfSoundings

    # print out progress every [printerval] soundings
    printerval = 10

    # ADD NEAREST MODEL-GRID LAT/LON VARIABLES
    # create empty arrays for lats/lons to be calculated
    NearDistance = np.full(np.shape(SoundingData.sounding), np.nan)
    NearLat      = np.full(np.shape(SoundingData.sounding), np.nan)
    NearLon      = np.full(np.shape(SoundingData.sounding), np.nan)
    NearAlt      = np.full(np.shape(SoundingData.lat), np.nan)
    NearAltDiff  = np.full(np.shape(SoundingData.lat), np.nan)
    # time is stored as a string
    NearTime = np.full(np.shape(SoundingData.sounding), str('yyyy-mm-dd hh:mm:ss'))
    # create arrays indices (-9 is the unfilled error value)
    NearNcoli     = np.full(np.shape(SoundingData.sounding), -9)
    NearTimei     = np.full(np.shape(SoundingData.sounding), -9)
    NearLevi      = np.full(np.shape(SoundingData.lat), -9)



    # FIND THE NEAREST LAT, LON AND TIME
    for i in range( StartSounding, EndSounding): # for every sounding in the array

        # print every printerval soundings
        if np.mod(i,printerval) == 0:
            print('Part 1 for sounding ' + str(i))


        # time from baloons, only consider the time at 1,000 m
        # or take the first time above 1,000 m that is not naT
        iii = 0
        while np.isnan( np.array(SoundingData.lat)[i][100+iii] ) and (iii < 2999) :
            iii = iii + 1



        # IF THERE ARE NO MEASUREMENTS ABOVE 1 KM, DO NOT INCLUDE THE SOUNDING!!!!!
        if (iii != 2999) :

            ExactTime   = (pd.Timestamp(str(np.array(SoundingData.flight_time)[i][100+iii])))
            NearTime[i] = str(ExactTime.round(freq='30T'))

            yyyy_mm_dd = str(np.array((NearTime[i])))[0:10]


            # retrieve month number and day of month from yyyy_mm_dd
            mmonth = int(yyyy_mm_dd[5:7])
            dday   = int(yyyy_mm_dd[8:10])

            # Day Of Year Calculation (only works in Jan/Feb otherwise overestimates)
            DOY_JF = ( (mmonth-1) * 31) + dday


            # ONLY ADD DATA TO THE XARRAY IF THE DATE OF NEAREST CESM TIME IS BETWEEN
            # 5 JANUARY AND 25 FEBRUARY (when we have data for) !!!!!!!!!!!!!!!!!!!!!
            if (DOY_JF >= 5) and (DOY_JF) <= 56:


                # Download CESM Output from Home Computer/Downloads (originally from group storage)
                # spefically for the day of this dropsonde launch
                CESMdata = xr.open_dataset\
                (VOLNAME+'/ThesisData/LargeDomainCESMoutputWithTh/' + xStr + '/LeadDay' + str(lead) + \
                 '/FHIST-ne30-ATOMIC-ERA5-' + xStr + '.cam.h3.' + yyyy_mm_dd +
                 '-00000WithTh.nc',engine='netcdf4')


                # lat and lon from baloons, only consider the lat and lon at 1,000 m
                # or take the first lat and lon above 1,000 m that is not Nan for Lat
                EURlat = np.array(SoundingData.lat)[i][100+iii]
                EURlon = np.array(SoundingData.lon)[i][100+iii]

                # set very large minimum distance to start
                MinDistance = 99999999
                MinDistLat = 99999999
                MinDistLon = 99999999

                for k in range(0,np.size(CESMdata.lat)):

                    # lat and lon of the real data
                    CESMlat = np.array(CESMdata.lat)[k]
                    CESMlon = np.array(CESMdata.lon)[k]


        #             print('')
        #             print('')
        #             print(CESMlat)
        #             print(CESMlon)
        #             print('')
        #             print(EURlat)
        #             print(EURlon)

                    # note theat LON IS IN 0-360 format (270 for 90 West)
                    # point distance in metres
                    Distance = GreatCircle(EURlat, EURlon, CESMlat, CESMlon)

                    # if the calculated distance is the new minimum:
                    if Distance < MinDistance:
                        MinDistance = Distance
                        MinDistLat  = CESMlat
                        MinDistLon  = CESMlon
                    # (near point)

                # arrays of near points/times
                NearLat[i]      = MinDistLat
                NearLon[i]      = MinDistLon
                NearDistance[i] = MinDistance


                # time from baloons, only consider the time at 1,000 m
                # or take the first time above 1,000 m that is not naN for Lat
                ExactTime   = (pd.Timestamp(str(np.array(SoundingData.flight_time)[i][100+iii])))
                NearTime[i] = str(ExactTime.round(freq='30T'))

                # ROUND TO THE NEAREST 30 MIN FOR .h3. DATA!!!
                # ROUND TO THE NEAREST 30 MIN FOR .h3. DATA!!!
                # ROUND TO THE NEAREST 30 MIN FOR .h3. DATA!!!
                # ROUND TO THE NEAREST 30 MIN FOR .h3. DATA!!!
                # ROUND TO THE NEAREST 30 MIN FOR .h3. DATA!!!
                # ROUND TO THE NEAREST 30 MIN FOR .h3. DATA!!!

        else:
            print('Sounding ' + str(i) + ' has no measurements above 1 km altitude')


    # CREATE XARRAY VARIABLES and populate them with the nearest lat/lon/time (and distance)
    NearLatXR = xr.DataArray(NearLat,dims=['sounding'],coords={'sounding':SoundingData.sounding.values}, \
                             name='lat_incesm')
    NearLatXR.attrs['long_name'] = 'latitude of nearest cesm point'
    NearLatXR.attrs['units'] = 'degrees_north'

    NearLonXR = xr.DataArray(NearLon,dims=['sounding'], \
                coords={'sounding':SoundingData.sounding.values},name='lon_incesm')
    NearLonXR.attrs['long_name'] = 'longitude of nearest cesm point'
    NearLonXR.attrs['units'] = 'degrees_east'

    NearDistXR = xr.DataArray(NearDistance,dims=['sounding'], \
                coords={'sounding':SoundingData.sounding.values},name='dist_incesm')
    NearDistXR.attrs['long_name'] = 'distance from nearest cesm point'
    NearDistXR.attrs['units'] = 'm'

    NearTimeXR = xr.DataArray(NearTime,dims=['sounding'], \
                  coords={'sounding':SoundingData.sounding.values},name='time_incesm')
    NearTimeXR.attrs['long_name'] = 'nearest cesm output time'

    SoundingData = xr.merge([SoundingData,NearLatXR,NearLonXR,NearDistXR,NearTimeXR])



    # number of altitude levels in sounding data, for computational speed
    NumOfSoundingLevs = np.size(SoundingData.alt)

    # FOR EACH, FIND THE NEAREST ALTITUDE
    for i in range( StartSounding, EndSounding):  # for every sounding in the array

        # print every printerval soundings
        if np.mod(i,printerval) == 0:
            print('Part 2 for sounding ' + str(i))

        # UNFORTUNAELY WE HAVE TO DOWNLOAD ALL OF THE CESM DATA AGAIN BECAUSE
        # THE CODE IS SET UP TO LOOP THROUGH EACH SOUNDING 3 TIMES

        # time from baloons, only consider the time at 1,000 m
        # or take the first time above 1,000 m that is not naN for Lat
        iii = 0
        while np.isnan( np.array(SoundingData.lat)[i][100+iii] ) and (iii < 2999) :
            iii = iii + 1

        # IF THERE ARE NO MEASUREMENTS ABOVE 1 KM, DO NOT INCLUDE THE SOUNDING!!!!!
        if (iii != 2999) :

            ExactTime   = (pd.Timestamp(str(np.array(SoundingData.flight_time)[i][100+iii])))
            NearTime[i] = str(ExactTime.round(freq='30T'))

            yyyy_mm_dd = str(np.array((NearTime[i])))[0:10]


            # retrieve month number and day of month from yyyy_mm_dd
            mmonth = int(yyyy_mm_dd[5:7])
            dday   = int(yyyy_mm_dd[8:10])

            # Day Of Year Calculation (only works in Jan/Feb otherwise overestimates)
            DOY_JF = ( (mmonth-1) * 31) + dday

            # ONLY ADD DATA TO THE XARRAY IF THE DATE OF NEAREST CESM TIME IS BETWEEN
            # 5 JANUARY AND 25 FEBRUARY (when we have data for) !!!!!!!!!!!!!!!!!!!!!
            if (DOY_JF >= 5) and (DOY_JF) <= 56:


                # Download CESM Output from Home Computer/Downloads (originally from group storage)
                # spefically for the day of this dropsonde launch
                CESMdata = xr.open_dataset\
                (VOLNAME+'/ThesisData/LargeDomainCESMoutputWithTh/' + xStr + '/LeadDay' + str(lead) + \
                 '/FHIST-ne30-ATOMIC-ERA5-' + xStr + '.cam.h3.' + yyyy_mm_dd +
                 '-00000WithTh.nc',engine='netcdf4')

                # find the "ncol" index for where the nearlat/ nearlon are in the model
                ncoli = CESMdata.ncol.where(CESMdata.lat == SoundingData.lat_incesm[i])
                ncoli = ncoli.where(CESMdata.lon == SoundingData.lon_incesm[i])
                ncoli = int(np.array(ncoli.dropna('ncol')))

                NearNcoli[i]  = int(ncoli)

                # find the time index for the nearest time in the model
                timei = (np.where(CESMdata.indexes['time'].to_datetimeindex() == \
                            str(np.array(SoundingData.time_incesm[i]))))
                timei = int(np.array(timei))

                NearTimei[i]  = int(timei)


                # retrieve an array of altitudes for this particular ncol and time
                CESMlocalAlts = np.array(CESMdata.Z3[timei,:,ncoli])

                # convert the altitude data to a local array for computational speed
                SoundingAlts  = np.array(SoundingData.alt)

                # number of altitude levels in CESMdata, for computational speed
                NumOfLevs = np.size(CESMdata.lev)



                for j in range(0, NumOfSoundingLevs):

                    # altitude of the real data
                    EURalt = SoundingAlts[j]

                    # set very large minimum difference to start
                    MinAltDiff    = 99999999
                    NearLeviDummy = -9         # error fill value for level index


                    # FOR FIRST SOUNDING LEVEL
                    if j == 0:
                        for k in range(0, NumOfLevs):
                            CESMalt = CESMlocalAlts[k]

                            # altitude difference in metres
                            AltDiff = np.abs( EURalt - CESMalt )
                            # if the calculated pressure difference is the new minimum:
                            if AltDiff < MinAltDiff:
                                MinAltDiff    = AltDiff
                                MinDistAlt    = CESMalt
                                NearLeviDummy = k

                        # store altitude-based "nearest" variables in arrays
                        NearAlt[i][j]     = MinDistAlt
                        NearAltDiff[i][j] = MinAltDiff
                        NearLevi[i][j]    = NearLeviDummy

                    # FOR ALL OTHER SOUNDING LEVELS
                    else:

                        # if the next sounding variable is not more than 1.5% different than
                        # the previous near alt, the near alt is the same (DEFINITELY)
                        PrevLevi = NearLevi[i][j-1]
                        PrevNearAlt = CESMlocalAlts[ PrevLevi ]
                        DiffFromPrevNearAlt = np.abs(EURalt - PrevNearAlt)
                        if DiffFromPrevNearAlt  < (0.015 * PrevNearAlt):

                            NearAlt[i][j]     = PrevNearAlt
                            NearAltDiff[i][j] = DiffFromPrevNearAlt
                            NearLevi[i][j]    = PrevLevi

                        # otherwise, go through every level to find min dist
                        else:
                            for k in range(0, NumOfLevs):

                                # particular altitude in CESM
                                CESMalt = CESMlocalAlts[k]

                                # altitude difference in metres
                                AltDiff = np.abs( EURalt - CESMalt )

                                # if the calculated pressure difference is the new minimum:
                                if AltDiff < MinAltDiff:
                                    MinAltDiff    = AltDiff
                                    MinDistAlt    = CESMalt
                                    NearLeviDummy = k

                            # store altitude-based "nearest" variables in arrays
                            NearAlt[i][j]     = MinDistAlt
                            NearAltDiff[i][j] = MinAltDiff
                            NearLevi[i][j]    = NearLeviDummy



    # CREATE XARRAY VARIABLES and populate them with the nearest altitudes (and distances)

    # nearest altitude level in model
    NearAltXR = xr.DataArray(NearAlt,dims=['sounding','alt'],coords=\
                             {'sounding':SoundingData.sounding.values,'alt':SoundingData.alt.values}, \
                             name='alt_incesm')
    NearAltXR.attrs['long_name'] = 'altitude of nearest cesm point'
    NearAltXR.attrs['units'] = 'm'


    # distance to nearest altitude level in model
    NearAltDiffXR = xr.DataArray(NearAltDiff,dims=['sounding','alt'],coords=\
                                 {'sounding':SoundingData.sounding.values,'alt':SoundingData.alt.values}, \
                                 name='altdiff_incesm')
    NearAltDiffXR.attrs['long_name'] = \
    'altitude difference from the nearest cesm point'
    NearAltDiffXR.attrs['units'] = 'm'


    # index in model of nearest column in model
    NearNcoliXR = xr.DataArray(NearNcoli,dims=['sounding'],coords=\
                               {'sounding':SoundingData.sounding.values}, \
                               name='ncol_ind_incesm')
    NearNcoliXR.attrs['long_name'] = \
    'corresponding index of ncol in cesm data'


    # index in model of nearest time in model
    NearTimeiXR = xr.DataArray(NearTimei,dims=['sounding'],coords=\
                               {'sounding':SoundingData.sounding.values}, \
                               name='time_ind_incesm')
    NearTimeiXR.attrs['long_name'] = \
    'corresponding index of time in cesm data'


    # index in model of nearest altitude in model
    NearLeviXR = xr.DataArray(NearLevi,dims=['sounding','alt'],coords=\
                              {'sounding':SoundingData.sounding.values,'alt':SoundingData.alt.values}, \
                              name='alt_ind_incesm')
    NearLeviXR.attrs['long_name'] = \
    'corresponding index of lev in cesm data'


    SoundingData= xr.merge([SoundingData, NearAltXR, NearAltDiffXR, NearTimeiXR, NearNcoliXR, NearLeviXR])



    # ADDING VARIABLES BY LINEAR INTERPOLATION SECTION
    for vari in range(0, np.size(SoundingVars) ):

        # retrieve specific variables from Sounding Xarray
        SoundingVar = xr.DataArray( SoundingData.variables[ SoundingVars[vari] ] )

        # local array version of the variable data for computational speed (not used in saving data)
        SoundingVarArray = np.array(SoundingVar)

        #create emptry array for storing model predictions
        CESMval   = np.full(np.shape(SoundingVar), np.nan)
        CESMerror = np.full(np.shape(SoundingVar), np.nan)


        # RETRIEVE MODEL PREDICTIONS
        for i in range( StartSounding, EndSounding): # for every sounding in the array

            TimeA = datetime.datetime.now()

            # print every printerval soundings
            if np.mod(i,printerval) == 0:
                print('Part ' + SoundingVars[vari] + ' for sounding '  + str(i) )

            TimeB = datetime.datetime.now()


            # ONLY ADD VARIABLE DATA TO THOSE SOUNDINGS THAT HAD CESM COORDINATES ADDED TO THEM
            if ~np.isnan(SoundingData.lat_incesm[i]) :

                TimeC = datetime.datetime.now()

                # retrieve month number and day of month from launch time
                yyyy_mm_dd = str(np.array(SoundingData.launch_time[i]))[0:10]
                mmonth = int(yyyy_mm_dd[5:7])
                dday   = int(yyyy_mm_dd[8:10])

                TimeD = datetime.datetime.now()

                # Day Of Year Calculation (only works in Jan/Feb otherwise overestimates)
                DOY_JF = ( (mmonth-1) * 31) + dday

                TimeE = datetime.datetime.now()

                # ONLY ADD DATA TO THE XARRAY IF THE DATE OF NEAREST CESM TIME IS BETWEEN
                # 5 JANUARY AND 25 FEBRUARY (when we have data for) !!!!!!!!!!!!!!!!!!!!!
                if (DOY_JF >= 5) and (DOY_JF) <= 56:

                    TimeF = datetime.datetime.now()

                    # Download CESM Output from Home Computer/Downloads (originally from group storage)
                    # spefically for the day of this dropsonde launch
                    CESMdata = xr.open_dataset\
                    (VOLNAME+'/ThesisData/LargeDomainCESMoutputWithTh/' + xStr + '/LeadDay' + str(lead) + \
                     '/FHIST-ne30-ATOMIC-ERA5-' + xStr + '.cam.h3.' + yyyy_mm_dd +
                     '-00000WithTh.nc',engine='netcdf4')

                    TimeG = datetime.datetime.now()

                    # retrieve specific variables from CESM Xarray
                    CESMvar      = xr.DataArray(CESMdata.variables[ CESMvars[vari] ] )
                    CESMvarArray = np.array(CESMvar)

                    TimeH = datetime.datetime.now()


                    # convert the altitude data to a local array for computational speed
                    SoundingAlts  = np.array(SoundingData.alt)
                    SoundingAltsC = np.array(SoundingData.alt_incesm)
                    CESMalts      = np.array(CESMdata.Z3)

                    # convert the index in cesm data to a local array for computational speed
                    SoundingTimesCI = np.array(SoundingData.time_ind_incesm)
                    SoundingNcolsCI = np.array(SoundingData.ncol_ind_incesm)
                    SoundingAltsCI  = np.array(SoundingData.alt_ind_incesm)

                    # the highest value that levi could be
                    HighLevi = np.shape(CESMdata.Z3)[1]-1

                    TimeI = datetime.datetime.now()


                    for j in range(0, np.shape(SoundingVar)[1]): # for each altitude in the sounding data (every 10 m)

                        if not SoundingAltsCI[i][j] == np.nan:

                            # Retrieve indices
                            timei = int(SoundingTimesCI[i])
                            ncoli = int(SoundingNcolsCI[i])
                            levi  = int(SoundingAltsCI[i][j])

                            # CALCULATE MODEL PREDICTIONS (LINEAR INTERPOLATION HERE)

                            # REMEMBER THAT CESM DATA IS BASED IN PRESSURE, SO LOW INDICES ARE HIGH ALTITUDES!!!!!

                            # if sounding data is above the highest model level or below the lowest level, then
                            # take just the highest lev or lowest lev variable value respectively


        #                     print('Sounding Altitude ' + str(SoundingAlts[j]))
        #                     print('Nearest Model Altitude ' + str(SoundingAltsC[i][j]))
        #                     print('Model Index ' + str(levi))
        #                     print('Higest Model Index ' + str(HighLevi))
        #                     print(' ')


                            # if sounding data is above (or equal) the nearest lev, calculate using the lev above it too (lower levi)
                            if SoundingAlts[j] >= SoundingAltsC[i][j] and levi != 0:
                                levi2 = levi - 1

                                ModelTopAlt = CESMalts[timei][levi2][ncoli]
                                BalloonAlt  = SoundingAlts[j]
                                ModelBotAlt = CESMalts[timei][levi][ncoli]

                                ModelAltDiff   = (ModelTopAlt - ModelBotAlt) # vertical distance between model levels
                                BalloonAltDiff = (BalloonAlt - ModelBotAlt)  # '' from lower model level to balloon
                                # how high above the lower model level is the balloon point (0.5 means halfway to higher point)
                                RelDist = ( BalloonAltDiff / ModelAltDiff )
                                # interpolate model value
                                CESMval[i][j]   = ( (1-RelDist) * CESMvarArray[timei, levi,ncoli] ) + \
                                                  (   (RelDist) * CESMvarArray[timei,levi2,ncoli] )

                            # if sounding data is below the nearest lev, calculate using the lev below it too (higher levi)
                            elif SoundingAlts[j] < SoundingAltsC[i][j] and levi != HighLevi:
                                levi2 = levi + 1

                                ModelTopAlt = CESMalts[timei][levi][ncoli]
                                BalloonAlt  = SoundingAlts[j]
                                ModelBotAlt = CESMalts[timei][levi2][ncoli]

                                ModelAltDiff   = (ModelTopAlt - ModelBotAlt)  # vertical distance between model levels
                                BalloonAltDiff = (BalloonAlt - ModelBotAlt)   # '' from lower model level to balloon
                                # how high above the lower model level is the balloon point
                                # (0.5 means halfway to higher point)
                                RelDist = ( BalloonAltDiff / ModelAltDiff )
                                # interpolate model value
                                CESMval[i][j]   = (     RelDist * CESMvarArray[timei,levi ,ncoli] ) + \
                                                  ( (1-RelDist) * CESMvarArray[timei,levi2,ncoli] )
                            # SPECIAL CASE FOR IF THE SOUNDING DATA IS ABOVE ALL MODEL LEVELS
                            elif levi == 0:
                                CESMval[i][j] = CESMvarArray[timei,levi,ncoli]

                            # SPECIAL CASE FOR IF THE SOUNDING DATA IS BELOW ALL MODEL LEVELS
                            elif levi == HighLevi:
                                CESMval[i][j] = CESMvarArray[timei,levi,ncoli]

                            else:
                                print('Altitude Error')
                                print('in sounding ' + str(i))
                                print('sounding altitude ' + str(SoundingAlts[j]))



                            # CALCULATE AND STORE MODEL ERRORS
                            CESMerror[i][j] = CESMval[i][j] - SoundingVarArray[i][j]



                    TimeJ = datetime.datetime.now()


#                     print('Time AB = ' + str(TimeB-TimeA) )
#                     print('Time BC = ' + str(TimeC-TimeB) )
#                     print('Time CD = ' + str(TimeD-TimeC) )
#                     print('Time DE = ' + str(TimeE-TimeD) )
#                     print('Time EF = ' + str(TimeF-TimeE) )
#                     print('Time FG = ' + str(TimeG-TimeF) )
#                     print('Time GH = ' + str(TimeH-TimeG) )
#                     print('Time HI = ' + str(TimeI-TimeH) )
#                     print('Time IJ = ' + str(TimeJ-TimeI) )
#                     print('Total Time = ' + str(TimeJ-TimeA) )
#                     print('')


                CESMdata.close()

        # ADD XARRAY VARIABLES TO DATA ARRAY FOR CESM VALUES AND CORRESPONDING ERRORS
        TScesmXR = xr.DataArray(CESMval,dims=['sounding','alt'],coords=\
                                {'sounding':SoundingData.sounding.values,'alt':SoundingData.alt.values}, \
                                name=str(SoundingVars[vari]) + '_cesmval')
# UNCOMMENT SECTION FOR THETA_L
#                                                                   name='thetal_cesmval')

        TScesmXR.attrs = SoundingVar.attrs
        TScesmXR.attrs['standard_name'] = \
        str(SoundingVar.attrs['standard_name']) + '_predicted_by_cesm'
        TScesmXR.attrs['long_name'] = \
        str(SoundingVar.attrs['long_name']) + ' predicted by cesm'

# UNCOMMENT SECTION FOR THETA_L
#         TScesmXR.attrs['standard_name'] = 'liquid_water_potential_temperature_predicted_by_cesm'
#         TScesmXR.attrs['long_name'] = 'liquid water potential temperature predicted by cesm'

        TSerrXR = xr.DataArray(CESMerror,dims=['sounding','alt'],coords=\
                               {'sounding':SoundingData.sounding.values,'alt':SoundingData.alt.values}, \
                               name=str(SoundingVars[vari]) + '_cesmerr')
# UNCOMMENT SECTION FOR THETA_L
#                                    name='thetal_cesmerr')
        TSerrXR.attrs = SoundingVar.attrs
        TSerrXR.attrs['standard_name'] = \
        str(SoundingVar.attrs['standard_name']) + '_error_in_cesm'
        TSerrXR.attrs['long_name'] = \
        str(SoundingVar.attrs['long_name']) + ' error in cesm'

# UNCOMMENT SECTION FOR THETA_L
#         TSerrXR.attrs = SoundingVar.attrs
#         TSerrXR.attrs['standard_name'] = 'liquid_water_potential_temperature_error_in_cesm'
#         TSerrXR.attrs['long_name'] = 'liquid water potential temperature error in cesm'

        SoundingData = xr.merge([SoundingData, TScesmXR, TSerrXR])


    # set final Xarray as the output
    SoundingDataPlus = SoundingData

    EndTime = datetime.datetime.now()

    print('')
    ThisTime = EndTime - StartTime
    TimePerUnit = ThisTime / ( (EndSounding-StartSounding) * 4)
    EstimatedTotalTime = TimePerUnit * 72000
    print('This Time = ' + str(ThisTime) )
    print('Time Per Unit = ' + str(TimePerUnit) )
    print('Estimated Total Time = ' + str(EstimatedTotalTime ) )

    return SoundingDataPlus


# THIS FUNCTION ADDS CESM OUTPUT VARIABLES TO ENHANCED SOUNDING DATA (BUT NO ERRORS)
# LINEAR INTERPOLATION IS USED

def AddMoreCESMvalues(EnhancedSoundingData, CESMvars):

    NumOfSoundings = np.size(EnhancedSoundingData.sounding)

    # take temperature as a sample variable from the sounding data to size other data properly
    SampleSoundingVar = EnhancedSoundingData.ta

    # The first and last soundings to loop over
    StartSounding = 0
    EndSounding = NumOfSoundings

    # ADDING VARIABLES BY LINEAR INTERPOLATION SECTION
    for vari in range(0, np.size(CESMvars) ):

        #create emptry array for storing model predictions
        CESMval   = np.full(np.shape(SampleSoundingVar), np.nan)
        CESMerror = np.full(np.shape(SampleSoundingVar), np.nan)


        # RETRIEVE MODEL PREDICTIONS
        for i in range(StartSounding, EndSounding): # for every sounding in the array

            # how often do you want to print out progress on soundings?
            printerval = 5
            # print every printerval soundings
            if np.mod(i,printerval) == 0:
                print('Part ' + CESMvars[vari] + ' for sounding '  + str(i) )

            # ONLY ADD VARIABLE DATA TO THOSE SOUNDINGS THAT HAD CESM COORDINATES ADDED TO THEM
            if ~np.isnan(EnhancedSoundingData.lat_incesm[i]) :

                # retrieve month number and day of month from launch time
                yyyy_mm_dd = str(np.array(EnhancedSoundingData.launch_time[i]))[0:10]
                mmonth = int(yyyy_mm_dd[5:7])
                dday   = int(yyyy_mm_dd[8:10])

                # Day Of Year Calculation (only works in Jan/Feb otherwise overestimates)
                DOY_JF = ( (mmonth-1) * 31) + dday


                # ONLY ADD DATA TO THE XARRAY IF THE DATE OF NEAREST CESM TIME IS BETWEEN
                # 5 JANUARY AND 25 FEBRUARY (when we have data for) !!!!!!!!!!!!!!!!!!!!!
                if (DOY_JF >= 5) and (DOY_JF) <= 56:

                    # Download CESM Output from Home Computer/Downloads (originally from group storage)
                    # spefically for the day of this dropsonde launch
                    CESMdata = xr.open_dataset\
                    (VOLNAME+'/DATA/LargeDomainCESMoutput/' + xStr + '/LeadDay' + str(lead) + \
                     '/FHIST-ne30-ATOMIC-ERA5-' + xStr + '.cam.h3.' + yyyy_mm_dd +
                     '-00000.nc',engine='netcdf4')

                    # retrieve specific variables from CESM Xarray
                    CESMvar      = xr.DataArray(CESMdata.variables[ CESMvars[vari] ] )
                    CESMvarArray = np.array(CESMvar)

                    # convert the altitude data to a local array for computational speed
                    SoundingAlts  = np.array(EnhancedSoundingData.alt)
                    SoundingAltsC = np.array(EnhancedSoundingData.alt_incesm)
                    CESMalts      = np.array(CESMdata.Z3)

                    # convert the index in cesm data to a local array for computational speed
                    SoundingTimesCI = np.array(EnhancedSoundingData.time_ind_incesm)
                    SoundingNcolsCI = np.array(EnhancedSoundingData.ncol_ind_incesm)
                    SoundingAltsCI  = np.array(EnhancedSoundingData.alt_ind_incesm)

                    # the highest value that levi could be
                    HighLevi = np.shape(CESMdata.Z3)[1]-1


                   # for each altitude in the sounding data (every 10 m)
                    for j in range(0, np.shape(SampleSoundingVar)[1]):

                        if not SoundingAltsCI[i][j] == np.nan:

                            # Retrieve indices
                            timei = int(SoundingTimesCI[i])
                            ncoli = int(SoundingNcolsCI[i])
                            levi  = int(SoundingAltsCI[i][j])

                            # CALCULATE MODEL PREDICTIONS (LINEAR INTERPOLATION HERE)

                            # REMEMBER THAT CESM DATA IS BASED IN PRESSURE, SO LOW INDICES ARE HIGH ALTITUDES!!!!!

                            # if sounding data is above the highest model level or below the lowest level, then
                            # take just the highest lev or lowest lev variable value respectively



                            # if sounding data is above (or equal) the nearest lev, calculate using the lev above it too (lower levi)
                            if SoundingAlts[j] >= SoundingAltsC[i][j] and levi != 0:
                                levi2 = levi - 1

                                ModelTopAlt = CESMalts[timei][levi2][ncoli]
                                BalloonAlt  = SoundingAlts[j]
                                ModelBotAlt = CESMalts[timei][levi][ncoli]

                                ModelAltDiff   = (ModelTopAlt - ModelBotAlt) # vertical distance between model levels
                                BalloonAltDiff = (BalloonAlt - ModelBotAlt)  # '' from lower model level to balloon
                                # how high above the lower model level is the balloon point (0.5 means halfway to higher point)
                                RelDist = ( BalloonAltDiff / ModelAltDiff )
                                # interpolate model value
                                CESMval[i][j]   = ( (1-RelDist) * CESMvarArray[timei, levi,ncoli] ) + \
                                                  (   (RelDist) * CESMvarArray[timei,levi2,ncoli] )

                            # if sounding data is below the nearest lev, calculate using the lev below it too (higher levi)
                            elif SoundingAlts[j] < SoundingAltsC[i][j] and levi != HighLevi:
                                levi2 = levi + 1

                                ModelTopAlt = CESMalts[timei][levi][ncoli]
                                BalloonAlt  = SoundingAlts[j]
                                ModelBotAlt = CESMalts[timei][levi2][ncoli]

                                ModelAltDiff   = (ModelTopAlt - ModelBotAlt)  # vertical distance between model levels
                                BalloonAltDiff = (BalloonAlt - ModelBotAlt)   # '' from lower model level to balloon
                                # how high above the lower model level is the balloon point
                                # (0.5 means halfway to higher point)
                                RelDist = ( BalloonAltDiff / ModelAltDiff )
                                # interpolate model value
                                CESMval[i][j]   = (     RelDist * CESMvarArray[timei,levi ,ncoli] ) + \
                                                  ( (1-RelDist) * CESMvarArray[timei,levi2,ncoli] )
                            # SPECIAL CASE FOR IF THE SOUNDING DATA IS ABOVE ALL MODEL LEVELS
                            elif levi == 0:
                                CESMval[i][j] = CESMvarArray[timei,levi,ncoli]

                            # SPECIAL CASE FOR IF THE SOUNDING DATA IS BELOW ALL MODEL LEVELS
                            elif levi == HighLevi:
                                CESMval[i][j] = CESMvarArray[timei,levi,ncoli]

                            else:
                                print('Altitude Error')
                                print('in sounding ' + str(i))
                                print('sounding altitude ' + str(SoundingAlts[j]))

                CESMdata.close()


        SampleCESMdata = xr.open_dataset\
                    (VOLNAME+'/DATA/LargeDomainCESMoutput/x001/LeadDay0' + \
                     '/FHIST-ne30-ATOMIC-ERA5-x001.cam.h3.2020-02-07' + '-00000.nc',engine='netcdf4')

        SampleCESMdataVar = xr.DataArray(SampleCESMdata.variables[ CESMvars[vari] ] )

        # ADD XARRAY VARIABLES TO DATA ARRAY FOR CESM VALUES AND CORRESPONDING ERRORS
        TScesmXR = xr.DataArray(CESMval,dims=['sounding','alt'],coords=\
                                {'sounding':EnhancedSoundingData.sounding.values,'alt':EnhancedSoundingData.alt.values}, \
                                name = SampleCESMdataVar.attrs['basename'] + '_cesmval')
        TScesmXR.attrs = SampleCESMdataVar.attrs
#         TScesmXR.attrs['standard_name'] = \
#         str(SampleCESMdataVar.attrs['standard_name']) + '_predicted_by_cesm'
        TScesmXR.attrs['long_name'] = \
        str(SampleCESMdataVar.attrs['long_name']) + ' predicted by cesm'


        EnhancedSoundingData = xr.merge([EnhancedSoundingData, TScesmXR])



    # set final Xarray as the output
    SoundingDataPlusPlus = EnhancedSoundingData

    return SoundingDataPlusPlus





# THIS BLOCK GOES AHEAD AND ADDS CESM DATA TO THE SOUNDING DATA

print("--> Adding CESM data to sounding data")

# DANGER (IGNORING WARNINGS)
# DANGER (IGNORING WARNINGS)
# DANGER (IGNORING WARNINGS)
# DANGER (IGNORING WARNINGS)
# DANGER (IGNORING WARNINGS)

# (used to ignore leap year calendar warning)
import warnings
warnings.filterwarnings('ignore')


# variables that you want to add
#SoundingVariables = ['ta','q','u','v', 'theta','p']
#CESMvariables = ['T','Q','U','V', 'THETAC','PMID']
SoundingVariables = ['ta','q','u','v', 'theta']
CESMvariables = ['T','Q','U','V', 'THETAC']

# ships and CESM configurations you want to deal with

                            # *BCO is Barbados Cloud Observatory (not a ship)
MissionNames = ['Atalante_Meteomodem', 'Atalante_Vaisala', 'BCO_Vaisala', \
                'Meteor_Vaisala', 'MS-Merian_Vaisala', 'RonBrown_Vaisala']

#xStrs = ['x101'] #, 'x101', 'x201', 'x202', 'x203', 'x204']
# CMZ lets use xStrs from above



for missi in range(0, np.size(MissionNames) ):
    MISSIONNAME = MissionNames[missi]

    for xStri in range(0, np.size(xStrs)):
        xStr = xStrs[xStri]

        for lead in range(0,3):
            print(' ')
            print(MISSIONNAME)
            print(xStr)
            print('Lead Day ' + str(lead))
            print('    ')

            # DOWNLOAD DATA (5 seperate files, 1 for each ship)
            # USING LEVEL 2 DATA BECAUSE IT WORKS, NOT BECAUSE I KNOW WHAT THE OTHER LEVELS DO!!!
            OriginalFolder = VOLNAME+'/DATA/StephanSoundings/OriginalDownloads/'
#             OriginalFolder = VOLNAME+'/ThesisData/StephanSoundings/WithMoreCESMdata/' + xStr + '/'
            OriginalFileNameSansNC = 'EUREC4A_' + MISSIONNAME + '-RS_L2_v3.0.0'
            SoundingDataPath = OriginalFolder + OriginalFileNameSansNC + '.nc'
#             SoundingDataPath = OriginalFolder + OriginalFileNameSansNC +  '-' + xStr + '_LeadDay' + \
#             str(lead) + 'wTurb.nc'

            SoundingData = xr.open_dataset( SoundingDataPath ,engine='netcdf4')


            # ADD CESM DATA TO THE ARRAY
            PlusData = AddCESMcoords(SoundingData, xStr, lead, SoundingVariables, CESMvariables)


            # SAVE THE AUGMENTED ARRAY AS A NETCDF FILE
            # make the directory to save it in if it doesn't already exist
            SaveFolder = VOLNAME+'/ThesisData/StephanSoundings/WithCESMdata/' + xStr + '/'
#             SaveFolder = VOLNAME+'/ThesisData/StephanSoundings/WithMoreCESMdataAndTh/' + xStr + '/'
            if not os.path.exists(SaveFolder):
                os.makedirs(SaveFolder)

            PlusData.to_netcdf(SaveFolder + OriginalFileNameSansNC + '-' + xStr + '_LeadDay' + str(lead) + \
                               'WithTh.nc')





# THIS BLOCK GOES AHEAD AND ADDS THE TURBULENCE VARIABLES TO THE CESM DATA AND SAVES THEM

print("--> Adding turbulence variables to CESM data")

# variables that you want to add to the enhanced sounding data
CESMvariables = ['upwp','vpwp','wp2','tau_zm','em','Lscale']

# strings for the model configurations to be looped over
#xStrs = ['x101'] #, 'x101', 'x201', 'x202', 'x203', 'x204']
#lets use xStrs from above

# names of the field campaign missions that change in the file names
MissionNames = ['Atalante_Meteomodem', 'Atalante_Vaisala' , 'BCO_Vaisala'     , \
                'Meteor_Vaisala'     , 'MS-Merian_Vaisala', 'RonBrown_Vaisala']

for xstri in range(0,np.size(xStrs)):
    xStr = xStrs[xstri]
    print('\n' + xStr)
    for missi in range(0, np.size(MissionNames)): #,np.size(MissionNames)):
        MissionName = MissionNames[missi]
        print('\n' + MissionName)
        for lead in range(0,3):
            print('\n' + 'Lead Day ' + str(lead) + '\n')


            OriginalFileNameSansNC = 'EUREC4A_' + MissionName +'-RS_L2_v3.0.0-' + xStr + '_LeadDay' + str(lead)
            OriginalFileName = OriginalFileNameSansNC + '.nc'

            EnhancedSoundingData = xr.open_dataset(VOLNAME+'/ThesisData/' + \
                                                   'StephanSoundings/WithCESMdata/' + \
                                   xStr + '/' + OriginalFileNameSansNC + 'WithTh' + '.nc', engine='netcdf4')

            SoundingPlusTurbData = AddMoreCESMvalues(EnhancedSoundingData, CESMvariables)


            # SAVE THE NEW AUGMENTED ARRAY AS A NETCDF FILE
            # make the directory to save it in if it doesn't already exist
            SaveFolder = VOLNAME+'/ThesisData/StephanSoundings/WithCESMdata/' + \
            xStr + '/'

            if not os.path.exists(SaveFolder):
                os.makedirs(SaveFolder)

#             OriginalFileName = 'EUREC4A_MS-Merian_Vaisala-RS_L2_v3.0.0.nc'


            SoundingPlusTurbData.to_netcdf(SaveFolder + OriginalFileNameSansNC + 'wTurb.nc')


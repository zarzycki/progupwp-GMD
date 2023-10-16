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


def cmz_print_filename(filename):
    #pass
    print("CMZ: ++++ The file name is: ", filename)

def cmz_writing_fig(figname,figid):
    print("CMZ: **** Writing at spot ", figid, " figure: ", figname)

def print_break():
    print("***********************************************************************************")

def get_arg(index):
    try:
        sys.argv[index]
    except IndexError:
        return ''
    else:
        return sys.argv[index]


#CMZ adding general volume path
VOLNAME=get_arg(1)
FILEOUTTYPE = 'pdf'  # png or pdf

if not bool(VOLNAME):
    print("Need to specify VOLNAME")
    quit()

# In[2]:


#CMZ
plot_fontsize=12

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
    StorageFolder = VOLNAME+'/ThesisData/StephanSoundings/WithCESMdata/x001/'
    dummyStorageFile = 'EUREC4A_' + MISSIONNAME + '-RS_L2_v3.0.0-x001_LeadDay0WithTh.nc'

    cmz_print_filename(StorageFolder + dummyStorageFile)
    dummySoundingDataPlus = xr.open_dataset( StorageFolder + dummyStorageFile, engine='netcdf4')

    # fill those empty arrays
    for soundi in range(0, np.size(dummySoundingDataPlus.sounding)):
        SoundingDateStrs[totalsoundi] = str(np.array(dummySoundingDataPlus.time_incesm[soundi]))[0:10]
        SoundingTimeInds[totalsoundi] = int(dummySoundingDataPlus.time_ind_incesm[soundi])
        SoundingNcolInds[totalsoundi] = int(dummySoundingDataPlus.ncol_ind_incesm[soundi])
        totalsoundi = totalsoundi + 1



# AND A FEW OTHER THINGS THAT JUST NEED TO BE DEFINED BEFORE THE REST OF THE CODE IS RUN

# I know ahead of time that there are 1546 total soundings
NumTotalSoundings = 1546

# retrieve the sizes of CAM output arrays
this_filename=VOLNAME+'/DATA/LargeDomainCESMoutput/x001/' + \
                       'LeadDay1/FHIST-ne30-ATOMIC-ERA5-x001.cam.h3.2020-01-06-00000.nc'
cmz_print_filename(this_filename)
dummyCAMoutput = xr.open_dataset(this_filename, engine='netcdf4')

NumTimes = np.size(dummyCAMoutput.time)
NumNcols = np.size(dummyCAMoutput.ncol)
NumLevs  = np.size(dummyCAMoutput.lev)
NumILevs = np.size(dummyCAMoutput.ilev)


# create empty arrays for all sounding profiles of wind, height, and momentum flux for each sounding column
Zprofiles     = np.full([NumTotalSoundings,  NumLevs], np.nan)

Uprofiles     = np.full([NumTotalSoundings,  NumLevs], np.nan)
UPWPprofiles  = np.full([NumTotalSoundings, NumILevs], np.nan)



xStrs = ['x001', 'x101', 'x201', 'x202', 'x203', 'x204', 'x301', 'x302', 'x303', 'x304']


# In[17]:


# # THIS BLOCK ONLY NEEDS TO BE RUN IF THESE STORED VARIABLES DO NOT ALREADY EXIST !!!!!
# # THIS BLOCK ONLY NEEDS TO BE RUN IF THESE STORED VARIABLES DO NOT ALREADY EXIST !!!!!
# # THIS BLOCK ONLY NEEDS TO BE RUN IF THESE STORED VARIABLES DO NOT ALREADY EXIST !!!!!
# # THIS BLOCK ONLY NEEDS TO BE RUN IF THESE STORED VARIABLES DO NOT ALREADY EXIST !!!!!
# # THIS BLOCK ONLY NEEDS TO BE RUN IF THESE STORED VARIABLES DO NOT ALREADY EXIST !!!!!

# # THIS BLOCK CREATES 1546 x NumLev Arrays for CAM U wind, Z height, and U'W' momentum flux for all soundings

## CMZ add if flag
cmz_calc_cg=False

if cmz_calc_cg:
    for xstri in range(0, len(xStrs)):
        xStr = xStrs[xstri]
        print('Updating dudz and keff profiles for '+xStr)

        # loop over all soundings
        for totalsoundi in range(0, len(SoundingTimeInds)):
            #print(totalsoundi)

            YYYYMMDD = SoundingDateStrs[totalsoundi]
            TimeInd = SoundingTimeInds[totalsoundi]
            NcolInd = SoundingNcolInds[totalsoundi]

            # do not include dates for which there is no CAM output
            if ( (YYYYMMDD != '2020-02-27') and (YYYYMMDD != '2020-02-28') and (YYYYMMDD != '2020-02-29') and \
                 (YYYYMMDD != '2020-03-01') and (YYYYMMDD != 'yyyy-mm-dd') ):

                CAMoutput = xr.open_dataset(VOLNAME+'/DATA/LargeDomainCESMoutput/' + xStr + '/LeadDay1/' +\
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

        np.save(VOLNAME+'/ThesisVariables/METEO600dUdZprofiles' + xStr + '.npy', dUdZprofiles)
        np.save(VOLNAME+'/ThesisVariables/METEO600KeffProfiles' + xStr + '.npy', KeffProfiles)


print_break()


# THIS BLOCK DOWNLOADS STORED PROFILES OF dU/dZ and K_eff and FINDS WHERE THERE ARE TRUE UPGRADIENT FLUXES

#set the threshold for how large dU/dZ has to be to really consider an upgradient flux
dUdZthreshold = 0.10 # [m/s per km]
# and convert to SI units ( m/s / m)
SIdUdZthreshold = 0.001 * dUdZthreshold


# where the profiles of dU/dZ and K_eff are stored
#DownloadPath = VOLNAME+'/VARIABLES/OriginalSoundingdProfiles/'
DownloadPath = VOLNAME+'/ThesisVariables/METEO600'

# retrieve these profiles for each CAM configuration and calculate where there are upgradient fluxes
# and where they occur only when dU/dZ is very small
for xStr in xStrs:

    exec("dUdZprofiles" + xStr + " = " + \
         "np.load(DownloadPath + 'dUdZprofiles" + xStr + ".npy')")
    cmz_print_filename(DownloadPath + 'dUdZprofiles' + xStr + '.npy')

    exec("KeffProfiles" + xStr + " = " + \
         "np.load(DownloadPath + 'KeffProfiles" + xStr + ".npy')")
    cmz_print_filename(DownloadPath + 'KeffProfiles' + xStr + '.npy')

    exec("LargedUdZmap" + xStr + "  = np.where( (np.abs(dUdZprofiles" + xStr + ")<SIdUdZthreshold) ,0,1)")
    exec("UpgradientMap" + xStr + " = np.where(KeffProfiles" + xStr + "<0,1,0)")

    exec("TrueUpgradientMap" + xStr + " = LargedUdZmap" + xStr + " * UpgradientMap" + xStr )

    exec("DualMap" + xStr + " = TrueUpgradientMap" + xStr + " + UpgradientMap" + xStr )

    # Print minimum "valid" dudz not filtered out
    print(xStr)
    exec("cmz_dudz = np.abs(dUdZprofiles" + xStr +")")
    exec("cmz_dual = DualMap" + xStr)
    actualvals = np.where(cmz_dual == 2,cmz_dudz,1000000.)
    print(np.nanmin(actualvals))
    # If 100000. that means there are no "true" Keff < 0 lines.

print_break()

# THIS BLOCK ACTUALLY CREATES PLOTS OF WHERE UPGRADIENT FLUXES OCCUR
# pressure level limits for the plot


# where do you want the top and bottoms of the plots in terms of pressure?
TopP = 600 # [hPa]
BotP = 1000 # [hPa]

#CMZ
my_dpi = 60  # 600 for publication

# plot maps of where there are upgradient fluxes in CAM output
xvals = np.arange(0,1546,1)
yvals = np.arange(0,59,1)

# rough pressure level estimates
iLevVals = np.array(dummyCAMoutput.ilev)

xmesh, ymesh = np.meshgrid(xvals, iLevVals)

xmesh = xmesh.transpose()
ymesh = ymesh.transpose()

xStrs = ['x001', 'x101', 'x201', 'x202', 'x203', 'x204', 'x301', 'x302', 'x303', 'x304', 'x101']

for xstri in range(0, len(xStrs)):
    xStr = xStrs[xstri]

    plt.figure(figsize=(20, 2.5), dpi=my_dpi)
    exec("plt.pcolormesh(xmesh,ymesh,DualMap" + xStr + ", shading='auto', cmap='afmhot_r')")

    # add vertical blue lines at the start of each mission
    for missi in range(0,NumMissions):
        plt.plot( [MissionStartSoundings[missi]-0.5, MissionStartSoundings[missi]-0.5], [BotP, TopP], \
                 color = 'goldenrod') # -0.5 so that all included soundings in the new mission are to the right

    # colour limit is 0 to 2 so that 0 is black, 1 is red, and 2 is white (from cmap='hot')
    plt.clim([0,2])
    plt.xlim([0,NumTotalSoundings])
#     plt.yscale('log')
    plt.ylim([BotP,TopP])

    # label the plot in the top left corner a), b), or c) for figures in LaTeX
    FigLabels = ['a', 'b', 'c', \
                 'd', 'e', 'b', \
                 'g', 'h', 'i', \
                 'j', 'k', 'l', \
                 'm', 'n', 'o', \
                 'p', 'q', 'a'  ]

#     if   (xStr == 'U'):
#         PlotLabelText = FigLabels[0 + xstri*3] + ')'
#     elif (VarStr == 'V'):
#         PlotLabelText = FigLabels[1 + xstri*3] + ')'
#     elif (VarStr == 'Hwind'):
#         PlotLabelText = FigLabels[2 + xstri*3] + ')'
#     else:
#         PlotLabelText = ''

    PlotLabelText = FigLabels[xstri-1]

    # place the label at sounding 6 and 600 hPa
    Xposition = 6
    Yposition = 670
    plt.text(Xposition,Yposition, PlotLabelText + ')',fontsize=21, color='k', weight='bold')


    plt.title("Locations of Upgradient u'w' for " + xStr, fontsize=24)
#     plt.xlabel('Sounding Index', fontsize=20)
    plt.ylabel('Estimated\nPressure Level\n[hPa]', fontsize=16)


    SaveFolder = VOLNAME+'/ThesisPlots/'
    if not os.path.exists(SaveFolder):
        os.makedirs(SaveFolder)

    savefigname=SaveFolder + 'UpgradientFluxMap' + xStr + PlotLabelText + 'SansXlabel.png'
    plt.savefig(savefigname, \
                         facecolor='w',dpi=my_dpi, bbox_inches='tight')
    #savefigname=SaveFolder + 'UpgradientFluxMap' + xStr + PlotLabelText + 'SansXlabel.eps'
    #plt.savefig(savefigname, \
    #                     facecolor='w',dpi=400, format='eps', bbox_inches='tight')
    cmz_writing_fig(savefigname,1)
    plt.close()



print_break()

# THIS BLOCK PACKAGES ALL OF THE RMSE DATA SAVED IN .NPY FILES INTO ARRAYS CONVENIENT FOR PLOTTING


# Which Hour Blocks are you retrieving RMSEs for?
HourBlockLength = 24
StartHourStr = '00'

# list of the model runs
# (don't use the last 3 when calculating turbulent errors since they are what the errors are relative too)
xStrs = ['x001','x101','x201','x202','x203', 'x204','x301','x302','x303', 'x304']

ChosenXstrs = ['x001','x101','x201','x202','x203', 'x204','x301','x302','x303', 'x304']

corr_xstri = np.full(np.size(ChosenXstrs), -1)
for chosexstri in range(0, np.size(ChosenXstrs)):
    for allxstri in range(0, np.size(xStrs)):
        if (ChosenXstrs[chosexstri] == xStrs[allxstri]):
            corr_xstri[chosexstri] = allxstri


# Retrieve altitudes to plot against


dummySoundingData = xr.open_dataset(VOLNAME+'/DATA/StephanSoundings/OriginalDownloads/' + \
                    'EUREC4A_Atalante_Vaisala-RS_L2_v3.0.0.nc', engine='netcdf4')
cmz_print_filename(VOLNAME+'/DATA/StephanSoundings/OriginalDownloads/' + \
                    'EUREC4A_Atalante_Vaisala-RS_L2_v3.0.0.nc')

alts = np.array(dummySoundingData.alt)


Vars         = [         'T' ,            'Q',               'U',               'V', \
                      'Hwind',       'theta']

VarLongNames = ['Temperature', 'Mixing Ratio', 'U-Wind Velocity', 'V-Wind Velocity', \
                'Horizontal Wind Speed', 'Potential Temperature']

VarUnitses   = [          'K',         'g/kg',             'm/s',             'm/s', \
                        'm/s',            'K']


TurbVars         =  ['UpWp', 'VpWp', 'Wp2', \
                     'TAU_zm', 'EM', 'LSCALE']

TurbVarLongNames = ['U-Wind Vertical Turbulent Flux','V-Wind Vertical Turbulent Flux','Vertical Wind Variance', \
                    'Momentum Time Scale',      'Turbulent Kinetic Energy', 'Turbulent Mixing Length' ]

TurbVarUnitses   = [ 'm\u00b2/s\u00b2', 'm\u00b2/s\u00b2', 'm\u00b2/s\u00b2', \
                     's', 'm\u00b2/s\u00b2',       'm']


# LOOPS FOR STATE VARIABLES
for vari in range(0, np.size(Vars)):
    Var = Vars[vari]
    print(Var)

    # create an array for storing RMSE at each model run, lead time, and altitude
    exec(Var +  "_RMSEsArray = np.full( [np.size(xStrs), 3, np.size(alts)], np.nan )" )
    exec(Var + "_BiasesArray = np.full( [np.size(xStrs), 3, np.size(alts)], np.nan )" )
    exec(Var +  "_MeansArray = np.full( [np.size(xStrs), 3, np.size(alts)], np.nan )" )


    for chosenxstri in range(0, np.size(ChosenXstrs)):
        xstri = corr_xstri[chosenxstri]
        xStr = xStrs[xstri]
        for lead in [1]: #range(0,3):
            #RMSEtempArray_name = VOLNAME+'/VARIABLES/' + str(HourBlockLength) + 'HourBlocks/' + \
            #            StartHourStr +'UTCstart/' + xStr + '/' + 'RMSEs/' + \
            #            'Model' + Var + 'RMSE' + xStr + 'LeadDay' + str(lead) + '_00UTCstart' + str(HourBlockLength) + \
            #            'HourBlocks.npy'
            RMSEtempArray_name = VOLNAME+'/ThesisVariables/' + \
                        'Model' + Var + 'RMSE' + xStr + 'LeadDay' + str(lead) + '_00UTCstart' + str(HourBlockLength) + \
                        'HourBlocks.npy'
            RMSEtempArray = np.load(RMSEtempArray_name)
            cmz_print_filename(RMSEtempArray_name)

            #BiasTempArray_name = VOLNAME+'/VARIABLES/' + str(HourBlockLength) + 'HourBlocks/' + \
            #            StartHourStr +'UTCstart/' + xStr + '/' + 'Biases/' + \
            #            'Model' + Var + 'Bias' + xStr + 'LeadDay' + str(lead) + '_00UTCstart' + str(HourBlockLength) + \
            #            'HourBlocks.npy'
            BiasTempArray_name = VOLNAME+'/ThesisVariables/' + \
                        'Model' + Var + 'Bias' + xStr + 'LeadDay' + str(lead) + '_00UTCstart' + str(HourBlockLength) + \
                        'HourBlocks.npy'
            BiasTempArray = np.load(BiasTempArray_name)
            cmz_print_filename(BiasTempArray_name)

            #MeanTempArray_name = VOLNAME+'/VARIABLES/' + str(HourBlockLength) + 'HourBlocks/' + \
            #            StartHourStr +'UTCstart/' + xStr + '/' + 'Means/' + \
            #            'Model' + Var + 'Mean' + xStr + 'LeadDay' + str(lead) + '_00UTCstart' + str(HourBlockLength) + \
            #            'HourBlocks.npy'
            MeanTempArray_name = VOLNAME+'/ThesisVariables/' + \
                        'Model' + Var + 'Mean' + xStr + 'LeadDay' + str(lead) + '_00UTCstart' + str(HourBlockLength) + \
                        'HourBlocks.npy'
            MeanTempArray = np.load(MeanTempArray_name)
            cmz_print_filename(MeanTempArray_name)

            # save the loaded array in its place in the larger arrays
            exec(Var +  "_RMSEsArray[xstri, lead, :] = RMSEtempArray[0,:]")
            exec(Var + "_BiasesArray[xstri, lead, :] = BiasTempArray[0,:]")
            exec(Var +  "_MeansArray[xstri, lead, :] = MeanTempArray[0,:]")




# LOOPS FOR TURBULENCE VARIABLES
for vari in range(0, np.size(TurbVars)):
    Var = TurbVars[vari]
    print(Var)

    # create an array for storing RMSE at each model run, lead time, and altitude
    #CMZ exec(Var +  "_RMSEsArray = np.full( [np.size(xStrs), 3, np.size(alts)], np.nan )" )
    #CMZ exec(Var + "_BiasesArray = np.full( [np.size(xStrs), 3, np.size(alts)], np.nan )" )
    exec(Var +  "_MeansArray = np.full( [np.size(xStrs), 3, np.size(alts)], np.nan )" )


    for xstri in range(0, np.size(xStrs)): # only non-nudged runs for turbulent variables
        xStr = xStrs[xstri]
        for lead in [1]: #range(0,3):
            #RMSEtempArray_name = VOLNAME+'/VARIABLES/' + str(HourBlockLength) + 'HourBlocks/' + \
            #            StartHourStr +'UTCstart/' + xStr + '/' + 'RMSEs/' + \
            #            'Model' + Var + '_RMSE' + xStr + 'LeadDay' + str(lead) + '_00UTCstart' + str(HourBlockLength) + \
            #            'HourBlocks.npy'
            #CMZ RMSEtempArray_name = VOLNAME+'/ThesisVariables/' + \
            #CMZ             'Model' + Var + '_RMSE' + xStr + 'LeadDay' + str(lead) + '_00UTCstart' + str(HourBlockLength) + \
            #CMZ             'HourBlocks.npy'
            #CMZ RMSEtempArray = np.load(RMSEtempArray_name)
            #CMZ cmz_print_filename(RMSEtempArray_name)

            #BiasTempArray_name = VOLNAME+'/VARIABLES/' + str(HourBlockLength) + 'HourBlocks/' + \
            #            StartHourStr +'UTCstart/' + xStr + '/' + 'Biases/' + \
            #            'Model' + Var + '_Bias' + xStr + 'LeadDay' + str(lead) + '_00UTCstart' + str(HourBlockLength) + \
            #            'HourBlocks.npy'
            #CMZ BiasTempArray_name = VOLNAME+'/ThesisVariables/' + \
            #CMZ             'Model' + Var + '_Bias' + xStr + 'LeadDay' + str(lead) + '_00UTCstart' + str(HourBlockLength) + \
            #CMZ             'HourBlocks.npy'
            #CMZ BiasTempArray = np.load(BiasTempArray_name)
            #CMZ cmz_print_filename(BiasTempArray_name)

            #MeanTempArray_name = VOLNAME+'/VARIABLES/' + str(HourBlockLength) + 'HourBlocks/' + \
            #            StartHourStr +'UTCstart/' + xStr + '/' + 'Means/' + \
            #            'Model' + Var + 'Mean' + xStr + 'LeadDay' + str(lead) + '_00UTCstart' + str(HourBlockLength) + \
            #            'HourBlocks.npy'
            MeanTempArray_name = VOLNAME+'/ThesisVariables/' + \
                        'Model' + Var + 'Mean' + xStr + 'LeadDay' + str(lead) + '_00UTCstart' + str(HourBlockLength) + \
                        'HourBlocks.npy'
            MeanTempArray = np.load(MeanTempArray_name)
            cmz_print_filename(MeanTempArray_name)

            # save the loaded array in its place in the larger arrays
            #CMZ exec(Var +  "_RMSEsArray[xstri, lead, :] = RMSEtempArray[0,:]")
            #CMZ exec(Var + "_BiasesArray[xstri, lead, :] = BiasTempArray[0,:]")
            exec(Var +  "_MeansArray[xstri, lead, :] = MeanTempArray[0,:]")


print_break()

plot_cm1=True
if plot_cm1:
    cm1_start_index = 1    # what "level" do we start plotting at (0 means plot all)
    cm1files = ['cm1-reference/avg.nc']
    cm1colors = ['gray']

    # Initialize arrays
    cm1_num_files = len(cm1files)
    cm1u = [None] * cm1_num_files
    cm1z = [None] * cm1_num_files
    cm1v = [None] * cm1_num_files
    cm1t = [None] * cm1_num_files
    cm1th = [None] * cm1_num_files
    cm1q = [None] * cm1_num_files
    cm1wsp = [None] * cm1_num_files
    cm1upwp = [None] * cm1_num_files
    cm1vpwp = [None] * cm1_num_files
    cm1vg = [None] * cm1_num_files
    cm1ug = [None] * cm1_num_files
    cm1wprof = [None] * cm1_num_files

    # Loop over files and load data into arrays
    for i, cm1file in enumerate(cm1files):
        cm1data = xr.open_dataset(cm1file, engine='netcdf4')
        cmz_print_filename(cm1file)
        cm1u[i] = cm1data.u.values[0,:,0,0]
        cm1z[i] = cm1data.zh.values[:]
        cm1z[i] = cm1z[i] / 1000.
        cm1v[i] = cm1data.v.values[0,:,0,0]
        cm1t[i] = cm1data.t.values[0,:,0,0]
        cm1th[i] = cm1data.th.values[0,:,0,0]
        cm1q[i] = cm1data.qv.values[0,:,0,0]
        cm1q[i] = cm1q[i] * 1000
        cm1wsp[i] = cm1data.wsp.values[0,:,0,0]
        cm1upwp[i] = cm1data.upwp.values[0,:,0,0]
        cm1vpwp[i] = cm1data.vpwp.values[0,:,0,0]
        cm1ug[i] = cm1data.ug.values[0,:,0,0]
        cm1vg[i] = cm1data.vg.values[0,:,0,0]
        cm1wprof[i] = cm1data.wprof.values[0,1::,0,0]

    # Create a dictionary to simplify plotting logic
    cm1_data_dict = {
        'U': cm1u,
        'V': cm1v,
        'T': cm1t,
        'Q': cm1q,
        'Hwind': cm1wsp,
        'theta': cm1th,
        'UpWp': cm1upwp,
        'VpWp': cm1vpwp
    }


# THIS BLOCK PLOTS RMSE or MEAN PROFILES FOR MANY RUNS/LEADS ON ONE PLOT

# THIS BLOCK NEEDS TO BE REWRITTEN TO BETTER ACCOMMODATE EASY SWITCHING BETWEEN VARS AND TURBVARS

# USER MODIFICATION SECTION!!!
# USER MODIFICATION SECTION!!!
# USER MODIFICATION SECTION!!!
# lowest and highest altitudes you want to plot for, and index in "alts" corresponding to those altitudes
arr_minAlt = [60,60,60,60,  60,60,60,60,  60,60,60,60]
arr_maxAlt = [2500,2500,5000,5000,  2500,2500,17000,17000,  2500,2500,17000,17000]
arr_titleOption = ['no','no','no','no',  'no','no','no','no',  'no','no','no','no']
arr_legendOption = ['no','inside','no',  'inside','no','inside','no','inside',  'inside','no','inside','no','inside']
arr_ncases = 12

for xx in range(arr_ncases):

    #CMZ edit this!
    if xx <= 3:
        ChosenXstrs  = ['x001', 'x101']
    elif xx > 3 and xx <= 7:
        ChosenXstrs  = ['x001', 'x101', 'x201', 'x202', 'x203', 'x204']
    else:
        ChosenXstrs  = ['x001', 'x101', 'x201', 'x202', 'x203', 'x204', 'x301', 'x302', 'x303', 'x304']

    print("DOING LOOP "+str(xx))

    minAlt  = arr_minAlt[xx] # [m]
    maxAlt  = arr_maxAlt[xx] # [m]

    # how do you want a title? (options = 'no', 'yes')
    TitleOption  = arr_titleOption[xx]

    # how do you want a legend? (options = 'no', 'left', 'inside', 'right')
    LegendOption = arr_legendOption[xx]

    minAlti = int(minAlt * 0.1)
    maxAlti = int( (maxAlt / 10) + 1 )

    AltsToPlot = alts[minAlti:maxAlti] * 0.001 # convert to [km]

    thickness = 1.8 # set line thicknesses in plot

    # set the name of the plots depending on if there is a title and a legend and where they are
    if (TitleOption == 'no'):
        AnnoStyle = 'SansTitle'
    elif(TitleOption == 'yes'):
        AnnoStyle = 'WithTitle'
    else:
        print("need a 'no' or 'yes' for Title Option")


    if (LegendOption == 'no'):
        AnnoStyle = AnnoStyle + 'SansLegend'

    elif(LegendOption == 'left'):
        AnnoStyle = AnnoStyle + 'LeftLegend'

    elif(LegendOption == 'inside'):
        AnnoStyle = AnnoStyle + 'InsideLegend'

    elif(LegendOption == 'right'):
        AnnoStyle = AnnoStyle + 'RightLegend'
    else:
        print("need a 'no' or 'left' or 'inside' or 'right' for Legend Option")


    # Variables to choose from
    Vars     =  [ 'T' , 'Q', 'U', 'V', 'Hwind', 'theta']
    # Vars     =  ['U', 'V', 'Hwind']
    # Vars = []
    #TurbVars =  ['UpWp', 'VpWp', 'Wp2', 'TAU_zm', 'EM', 'LSCALE']
    TurbVars = []


    # KEEP THIS LIST AS ALL XSTRINGS
    xStrs = ['x001', 'x101', 'x201', 'x202', 'x203', 'x204', 'x301', 'x302', 'x303', 'x304']

    # colours associated with each of the model runs
    PlotxStrColours = [ [0.9,0.0,0.0],
                        [0.0,0.5,0.0], \
                        [0.0,0.85,1.0], [0.0,0.6,1.0], [0.0,0.0,0.9], [0.4,0.0,0.5], \
                        [0.5,0.85,1.0], [0.5,0.6,1.0], [0.5,0.0,0.9], [0.4,0.5,0.5] ]

    # store the x-limits in the plots of each variable

    # FOR 2.5 KM TOP
    if (maxAlt == 2500):
                   #[  T,    Q,    U,    V,     H,     Th]
        MEANxmins = [284,    0,  -11, -3.0,     5,    297]
        MEANxmaxs = [300,   17,   -4,    0,    11,    310]

        BIASxmins = [-1.5,-1.2, -1.7, -1.7,  -1.3,  -1.75]
        BIASxmaxs = [ 0.0, 1.2,  1.7,  1.7,   1.3,      0]

        RMSExmins = [ 0.0, 0.0, 1.25, 1.25,   0.0,    0.0]
        RMSExmaxs = [ 2.0, 3.5, 3.25, 3.25,   3.0,   2.25]


    # FOR 5 KM TOP
    if (maxAlt == 5000):
                   #[   T,   Q,    U,    V,     H,     Th]
        MEANxmins = [272,    0,  -11, -3.0,     5,    297]
        MEANxmaxs = [298,   16,    0,    0,    11,    322]

        BIASxmins = [-1.5,-1.2, -1.7, -1.7,  -1.3,  -1.75]
        BIASxmaxs = [ 0.0, 1.2,  1.7,  1.7,   1.3,      0]

        RMSExmins = [ 0.0, 0.0, 1.25, 1.25,   0.0,    0.0]
        RMSExmaxs = [ 2.0, 3.5, 3.75, 3.75,   4.0,   2.25]


    # FOR 10 KM TOP
    if (maxAlt == 10000):
                   #[   T,   Q,    U,    V,     H,     Th]
        MEANxmins = [272,    0,  -11, -3.0,     5,    297]
        MEANxmaxs = [298,   16,    0,    0,    11,    322]

        BIASxmins = [-1.5,-1.2, -1.7, -1.7,  -1.3,  -1.75]
        BIASxmaxs = [ 0.0, 1.2,  1.7,  1.7,   1.3,      0]

        RMSExmins = [ 0.0, 0.0, 1.25, 1.25,   0.0,    0.0]
        RMSExmaxs = [ 2.0, 3.5, 3.75, 3.75,   4.0,   2.25]


    # FOR 17.5 KM TOP
    if (maxAlt == 17500) or (maxAlt == 8000):
                   #[   T,   Q,    U,    V,     H,    Th]
        MEANxmins = [190,    0,  -12,  -12,     0,   295]
        MEANxmaxs = [305, 16.5,   30,    2,  32.5,   400]

        BIASxmins = [-1.5,-1.2, -1.7, -1.7,  -1.3,  -1.75]
        BIASxmaxs = [ 0.0, 1.2,  1.7,  1.7,   1.3,      0]

        RMSExmins = [ 0.0, 0.0, 1.25, 1.25,   0.0,    0.0]
        RMSExmaxs = [ 2.0, 3.5, 3.75, 3.75,   4.0,   2.25]


    # FOR TURBULENT VARIABLES
               #[      UpWp,     VpWp,    Wp2,    TAU_zm,    EM,    LSCALE]
    # MEANxmins = [   -0.0225,   -0.023,  -0.02,     -0.02, -0.02,     -0.02]
    # MEANxmaxs = [    0.0625,    0.011,   0.09,      0.09,  0.09,      0.09]

    # MEANxmins = [-0.02,-0.03]
    # MEANxmaxs = [ 0.12, 0.03]



    # lead   =  0     # Choose which lead day you want plots for
    # Var    = 'Hwind'  # Choose the   variable you want plots for

    # CREATE AN ARRAY WHICH LISTS THE CORRESPONDING INDICES OF THE XSTRS CHOSEN
    # IN THE ORIGINAL MASTER XSTR LIST (this is here because np.find doesn't work)
    # create an empty array, must be filled with -1 to be an integer
    corr_xstri = np.full(np.size(ChosenXstrs), -1)

    for chosexstri in range(0, np.size(ChosenXstrs)):
        for allxstri in range(0, np.size(xStrs)):
            if (ChosenXstrs[chosexstri] == xStrs[allxstri]):
                corr_xstri[chosexstri] = allxstri

    for Var in Vars + TurbVars:

        # retrieve the index of the chosen variable within its respective array (for labelling purposes)
        if (Var in Vars):
            for i in range(0, np.size(Vars)):
                if (Var == Vars[i]):
                    vari = i

        if (Var in TurbVars):
            for i in range(0, np.size(TurbVars)):
                if (Var == TurbVars[i]):
                    turbvari = i


        for chosexstri in range(0, np.size(ChosenXstrs)):
            for allxstri in range(0, np.size(xStrs)):
                if (ChosenXstrs[chosexstri] == xStrs[allxstri]):
                    corr_xstri[chosexstri] = allxstri

    #     # set the x-limits on the plots
    #     RMSExmin = RMSExmins[vari]
    #     RMSExmax = RMSExmaxs[vari]
    #     BIASxmin = BIASxmins[vari]
    #     BIASxmax = BIASxmaxs[vari]
    #     MEANxmin = MEANxmins[vari]
    #     MEANxmax = MEANxmaxs[vari]
    #     # set the x-limits on the plots
    #     RMSExmin = RMSExmins[turbvari]
    #     RMSExmax = RMSExmaxs[turbvari]
    #     BIASxmin = BIASxmins[turbvari]
    #     BIASxmax = BIASxmaxs[turbvari]
    #     MEANxmin = MEANxmins[turbvari]
    #     MEANxmax = MEANxmaxs[turbvari]


    #     set the x-limits on the plots

        if (Var in Vars):
            RMSExmin = RMSExmins[vari]
            RMSExmax = RMSExmaxs[vari]
            BIASxmin = BIASxmins[vari]
            BIASxmax = BIASxmaxs[vari]
            MEANxmin = MEANxmins[vari]
            MEANxmax = MEANxmaxs[vari]

        if (Var in TurbVars):
            MEANxmin = MEANxmins[turbvari]
            MEANxmax = MEANxmaxs[turbvari]


        # count the number of nudged runs included by the user
        NumOfNudgedRuns = 0
        if ('x501' in ChosenXstrs):
            NumOfNudgedRuns = NumOfNudgedRuns + 1
        if ('x511' in ChosenXstrs):
            NumOfNudgedRuns = NumOfNudgedRuns + 1
        if ('x521' in ChosenXstrs):
            NumOfNudgedRuns = NumOfNudgedRuns + 1

        # if the variable is a turbulence variable, don't include the nudged runs
        if (Var in Vars):
            LastXstr = np.size(ChosenXstrs)
        elif(Var in TurbVars):
            LastXstr = np.size(ChosenXstrs) - NumOfNudgedRuns


        for lead in [1]: #range(0,3):
            # PLOTTING SECTION


    #         ONLY LOOP OVER BIASES AND RMSEs FOR NON-TURBULENCE VARIABLES
            if (Var in Vars):
                # RMSEs SECTION
                # plot the actual lines
                ax = plt.figure()
                runningRMSEmax = 0 # set starting maximum to 0
                local_xstri = -1  # set starting local loop index

                LastXstr = np.size(ChosenXstrs)
                for xStr in ChosenXstrs[0:LastXstr] :
                    local_xstri = local_xstri + 1
                    # retrieve the index of the chosen xStr in the master xStr list for plotting purposes
                    xstri = corr_xstri[local_xstri]

                    # retrievet the RMSE storage array for this variable
                    exec("RMSEsToPlot = " + Var + "_RMSEsArray[xstri,lead,0:maxAlti]")
        #             select only the altitudes beneath the chosen macumum
    #                 AltsToPlot = alts[0:500] * 0.001 # convert to [km]


                    # PLOT THE ACTUAL LINE
                    if (xStr == 'x001'):
                        plt.plot(RMSEsToPlot[minAlti:maxAlti], AltsToPlot, color = PlotxStrColours[xstri], \
                                 linewidth=thickness, linestyle=':', label=xStr)
                    else:
                        plt.plot(RMSEsToPlot[minAlti:maxAlti], AltsToPlot, color = PlotxStrColours[xstri], \
                                 linewidth=thickness, linestyle='--', label=xStr)

                    # keep track of the largest RMSE value on the plot
                    thisRMSEmax = np.nanmax(RMSEsToPlot)
                    if (thisRMSEmax > runningRMSEmax):
                        runningRMSEmax = thisRMSEmax

                RMSEmax = runningRMSEmax # set maximum to whatever the last running maximum was


                # plot accessories
                plt.grid('on', linewidth=0.25)

                # add a legend in the corresponding spot to user choice
                if(LegendOption == 'left'):
                    plt.legend(ChosenXstrs[0:LastXstr], bbox_to_anchor=(0.03, 0.9))
    #                 ax.legend(bbox_to_anchor=(0.03, 0.9))

                elif(LegendOption == 'inside'):
                    plt.legend(ChosenXstrs[0:LastXstr])

                elif(LegendOption == 'right'):
                    plt.legend(ChosenXstrs[0:LastXstr], bbox_to_anchor=(1.07, 0.9))
    #                 ax.legend(bbox_to_anchor=(1.07, 0.9))


                # label with the corresponding variable name whether its a state or a turbulence variable
                # only add a title if the user chooses it
                if  (Var in Vars):
                    if(TitleOption == 'yes'):
                        plt.title(str(lead) + '-Day Lead Vertical RMSE Profiles for ' + VarLongNames[vari])

                    plt.xlabel('Root Mean Squared Error of ' + VarLongNames[vari] + ' [' + VarUnitses[vari] + ']',fontsize=plot_fontsize)

                elif (Var in TurbVars):
                    if(TitleOption == 'yes'):
                        plt.title(str(lead) + '-Day Lead Vertical RMSE Profiles for ' + TurbVarLongNames[turbvari])

                    plt.xlabel('Root Mean Squared Error of ' + TurbVarLongNames[turbvari] + \
                               ' [' + TurbVarUnitses[turbvari] + ']',fontsize=plot_fontsize)

                plt.ylabel('Altitude [km]',fontsize=plot_fontsize)

        #         plt.xlim([0,RMSEmax*1.5]) # set the x range from 0 to 150% maximum
                plt.xlim([RMSExmin,RMSExmax])
                plt.ylim([0,maxAlt*0.001])


        #         # include a thin red 2.5 km line (for where the integrated calculations are cutoff)
        #         plt.plot([RMSExmin,RMSExmax],[2500,2500],'r', linewidth=0.75)

                SaveFolder = VOLNAME+'ThesisPlots/'

                if not os.path.exists(SaveFolder):
                    os.makedirs(SaveFolder)

                savefigname=SaveFolder + Var + 'RMSEprofiles' + str(lead) + 'DayLead' + \
                            ChosenXstrs[0] + ChosenXstrs[LastXstr-1] + '_' + str(maxAlt) + \
                            'm' + AnnoStyle + '.'+FILEOUTTYPE
                plt.savefig( savefigname , \
                            facecolor='w',dpi=300, bbox_inches='tight')
                cmz_writing_fig(savefigname,2)
                plt.close()





                # BIASES SECTION
                # plot the actual lines
                ax = plt.figure()
                runningBiasMax = 0 # set starting maximum to 0
                local_xstri = -1  # set starting local loop index

                for xStr in ChosenXstrs[0:LastXstr] :
                    local_xstri = local_xstri + 1
                    # retrieve the index of the chosen xStr in the master xStr list for plotting purposes
                    xstri = corr_xstri[local_xstri]

                    # retrievet the RMSE storage array for this variable
                    exec("BiasesToPlot = " + Var + "_BiasesArray[xstri,lead,0:maxAlti]")
                    # select only the altitudes beneath the chosen macumum
        #             AltsToPlot = alts[0:maxAlti]


                    # PLOT THE ACTUAL LINE
                    if (xStr == 'x001'):
                        plt.plot(BiasesToPlot[minAlti:maxAlti], AltsToPlot, color = PlotxStrColours[xstri], \
                                 linewidth=thickness, linestyle=':', label=xStr)
                    else:
                        plt.plot(BiasesToPlot[minAlti:maxAlti], AltsToPlot, color = PlotxStrColours[xstri], \
                                 linewidth=thickness, linestyle='--', label=xStr)

                    # keep track of the largest RMSE value on the plot
                    thisBiasMax = np.nanmax(np.abs(BiasesToPlot))
                    if (thisBiasMax > runningBiasMax):
                        runningBiasMax = thisBiasMax

                BiasMax = runningBiasMax # set maximum to whatever the last running maximum was


                # plot accessories
                plt.grid('on', linewidth=0.25)


                # add a legend in the corresponding spot to user choice
                if(LegendOption == 'left'):
                    plt.legend(ChosenXstrs[0:LastXstr], bbox_to_anchor=(0.03, 0.9))
    #                 ax.legend(bbox_to_anchor=(0.03, 0.9))

                elif(LegendOption == 'inside'):
                    plt.legend(ChosenXstrs[0:LastXstr],loc='lower right')

                elif(LegendOption == 'right'):
                    plt.legend(ChosenXstrs[0:LastXstr], bbox_to_anchor=(1.07, 0.9))
    #                 ax.legend(bbox_to_anchor=(1.07, 0.9))


                # label with the corresponding variable name whether its a state or a turbulence variable
                # only add a title if the user chooses it
                if  (Var in Vars):
                    if(TitleOption == 'yes'):
                        plt.title(str(lead) + '-Day Lead Vertical Bias Profiles for ' + VarLongNames[vari])

                    plt.xlabel('Mean Bias of ' + VarLongNames[vari] + ' [' + VarUnitses[vari] + ']',fontsize=plot_fontsize)
                elif (Var in TurbVars):
                    if(TitleOption == 'yes'):
                        plt.title(str(lead) + '-Day Lead Vertical Bias Profiles for ' + TurbVarLongNames[turbvari])

                    plt.xlabel('Mean Bias of ' + TurbVarLongNames[turbvari] + \
                               ' [' + TurbVarUnitses[turbvari] + ']',fontsize=plot_fontsize)

                plt.ylabel('Altitude [km]',fontsize=plot_fontsize)

        #         plt.xlim([BiasMax*-1.5,BiasMax*1.5]) # set the x range from 0 to 150% maximum
                plt.xlim([BIASxmin,BIASxmax])
                plt.ylim([0,maxAlt*0.001])


        #         # include a thin red 2.5 km line (for where the integrated calculations are cutoff)
        #         plt.plot([BIASxmin,BIASxmax],[2500,2500],'r', linewidth=0.75)

                # include a thin black y-axis line
                plt.plot([0,0],[0,maxAlt],'k', linewidth=0.5)


                SaveFolder = VOLNAME+'ThesisPlots/'

                if not os.path.exists(SaveFolder):
                    os.makedirs(SaveFolder)

                savefigname=SaveFolder + Var + 'BiasProfiles' + str(lead) + 'DayLead' + \
                            ChosenXstrs[0] + ChosenXstrs[LastXstr-1] + '_' + str(maxAlt) + \
                            'm' + AnnoStyle + '.'+FILEOUTTYPE
                plt.savefig( savefigname, \
                            facecolor='w',dpi=300, bbox_inches='tight')
                cmz_writing_fig(savefigname,3)
                plt.close()


            # MEANS SECTION

            # select only the altitudes beneath the chosen maximum
    #         AltsToPlot = alts[0:maxAlti]

    #       retrieve the means from observations for non-turbulence variables
            if  (Var in Vars):
                if Var == "Hwind":
                    print("special Hwind treatment!")
                    tmp1name=VOLNAME+'/ThesisVariables/' + \
                    'Obs' + 'U' + 'Mean_00UTCstart24HourBlocks.npy'
                    tmp2name=VOLNAME+'/ThesisVariables/' + \
                    'Obs' + 'V' + 'Mean_00UTCstart24HourBlocks.npy'
                    cmz_print_filename(tmp1name)
                    cmz_print_filename(tmp2name)
                    tmp1 = np.load(tmp1name)
                    tmp2 = np.load(tmp2name)
                    ObsMeans = np.sqrt(tmp1**2 + tmp2**2)
                else:
                    ObsMeans_name = VOLNAME+'/ThesisVariables/' + \
                    'Obs' + Var + 'Mean_00UTCstart24HourBlocks.npy'
                    ObsMeans = np.load(ObsMeans_name)
                    cmz_print_filename(ObsMeans_name)

            # take only the bottom 501 entries
            ObsMeansToPlot = np.reshape(ObsMeans,[3100])[0:maxAlti]

            ax = plt.figure()

            # PLOT A THICK BLACK LINE FOR THE OBSERVATIONS IN NON-TURBULENCE VARIABLES
            if  (Var in Vars):
                plt.plot(ObsMeansToPlot[minAlti:maxAlti], AltsToPlot, color = 'k', linewidth=2.0, \
                         label='Observations')

            if plot_cm1:
                # Check if Var exists in dictionary
                if Var in cm1_data_dict:
                    # Plot data for each file
                    for i, datatmp in enumerate(cm1_data_dict[Var]):
                        plt.plot(datatmp[cm1_start_index::], cm1z[i][cm1_start_index::], color=cm1colors[i % len(cm1colors)], linewidth=1.5, label='CM1')
                else:
                    print(f"CM1 variable '{Var}' not recognized.")

            # retrieve the means from model output for each run
            local_xstri = -1  # set starting local loop index
            for xStr in ChosenXstrs[0:LastXstr] :
                local_xstri = local_xstri + 1
                # retrieve the index of the chosen xStr in the master xStr list for plotting purposes
                xstri = corr_xstri[local_xstri]

                if Var == "Hwind":
                    exec("MeansToPlot = np.sqrt(U_MeansArray[xstri,lead,0:maxAlti]**2 + V_MeansArray[xstri,lead,0:maxAlti]**2)")
                    print("special hwind model treatment!")
                else:
                    exec("MeansToPlot = " + Var + "_MeansArray[xstri,lead,0:maxAlti]")

                # PLOT THE ACTUAL LINE
                if (xStr == 'x001'):
                    plt.plot(MeansToPlot[minAlti:maxAlti], AltsToPlot, linestyle=':', \
                             color = PlotxStrColours[xstri], linewidth=thickness, label=xStr)
                else:
                    plt.plot(MeansToPlot[minAlti:maxAlti], AltsToPlot, linestyle='--', \
                             color = PlotxStrColours[xstri], linewidth=thickness, label=xStr)

            # plot accessories
            plt.grid('on', linewidth=0.25)

            # label with the corresponding variable name whether its a state or a turbulence variable
            # include a legend with 'observations' only for non-turbulence variables
            # add a title and/or legend depending on user choice (with legend in location of user choice)
            print(Var)
            if  (Var in Vars):
                plt.xlabel('Mean ' + VarLongNames[vari] + ' [' + VarUnitses[vari] + ']',fontsize=plot_fontsize)

                if(TitleOption == 'yes'):
                    plt.title(str(lead) + '-Day Lead Vertical Mean Profiles for ' + VarLongNames[vari])

                special_strings = 'Observations'

                # If plotting CM1, glue that into the special strings array
                if plot_cm1:
                    special_strings = np.append(special_strings,'CM1')

                if(LegendOption=='left'):
                    ax.legend((np.append(special_strings, ChosenXstrs[0:LastXstr])), bbox_to_anchor=(0.03, 0.9))
                elif(LegendOption=='inside'):
                    plt.legend((np.append(special_strings, ChosenXstrs[0:LastXstr])),loc='lower right')
                elif(LegendOption=='right'):
                    ax.legend((np.append(special_strings, ChosenXstrs[0:LastXstr])), bbox_to_anchor=(1.17, 0.9))

            elif (Var in TurbVars):
                plt.xlabel('Mean ' + TurbVarLongNames[turbvari]  + ' [' + TurbVarUnitses[turbvari] + ']',fontsize=plot_fontsize)

                if(TitleOption == 'yes'):
                    plt.title(str(lead) + '-Day Lead Vertical Mean Profiles for ' + TurbVarLongNames[turbvari])

                if(LegendOption=='left'):
                    ax.legend(ChosenXstrs[0:LastXstr], bbox_to_anchor=(0.03, 0.9))
                elif(LegendOption=='inside'):
                    plt.legend(ChosenXstrs[0:LastXstr],loc='lower right')
                elif(LegendOption=='right'):
                    ax.legend(ChosenXstrs[0:LastXstr], bbox_to_anchor=(1.17, 0.9))


            plt.ylabel('Altitude [km]',fontsize=plot_fontsize)

            plt.xlim([MEANxmin,MEANxmax])
            plt.ylim([0,maxAlt*0.001])


    #         # include a thin red 2.5 km line (for where the integrated calculations are cutoff)
    #         plt.plot([BIASxmin,BIASxmax],[2500,2500],'r', linewidth=0.75)

            # include a thin black y-axis line
            plt.plot([0,0],[0,maxAlt],'k', linewidth=0.5)

            SaveFolder = VOLNAME+'/ThesisPlots/'

            if not os.path.exists(SaveFolder):
                os.makedirs(SaveFolder)

            savefigname=SaveFolder + Var + 'MeanProfiles' + str(lead) + 'DayLead' + \
                        ChosenXstrs[0] + ChosenXstrs[LastXstr-1] + '_' + str(maxAlt) + \
                        'm' + AnnoStyle + '.'+FILEOUTTYPE
            plt.savefig( savefigname, \
                        facecolor='w',dpi=300, bbox_inches='tight')
            cmz_writing_fig(savefigname,4)
            plt.close()


print_break()



# THIS BLOCK PLOTS MEAN PROFILES FOR TURBULENCE VARIABLES FROM MANY RUNS/LEADS ON ONE PLOT

# Variables to choose from

arr_minAlt = [60,60,  60,60,  60,60]
arr_maxAlt = [2500,2500,  2500,2500,  2500,2500]
arr_titleOption = ['no','no',  'no','no',  'no','no']
#arr_legendOption = ['no','inside','no','inside']
arr_ncases = 6

for xx in range(arr_ncases):

    maxAlt =  arr_maxAlt[xx]

    #CMZ edit this!
    if xx <= 1:
        ChosenXstrs  = ['x001', 'x101']
    elif xx > 1 and xx <= 3:
        ChosenXstrs  = ['x001', 'x101', 'x201', 'x202', 'x203', 'x204']
    else:
        ChosenXstrs  = ['x001', 'x101', 'x201', 'x202', 'x203', 'x204', 'x301', 'x302', 'x303', 'x304']

    print("DOING LOOP "+str(xx))

    Vars = [] #['ThetaL']
    TurbVars = ['UpWp', 'VpWp'] #,'Wp2', 'TAU_zm', 'EM', 'LSCALE']

    TurbVarUnitses   = [ 'm\u00b2/s\u00b2', 'm\u00b2/s\u00b2'] #, 'm\u00b2/s\u00b2', \
                         # 's', 'm\u00b2/s\u00b2',       'm']

    # colours associated with each of the model runs
    PlotxStrColours = [ [0.9,0.0,0.0], [0.0,0.5,0.0], \
                        [0.0,0.85,1.0], [0.0,0.6,1.0], [0.0,0.0,0.9], [0.4,0.0,0.5], \
                        [0.5,0.85,1.0], [0.5,0.6,1.0], [0.5,0.0,0.9], [0.4,0.5,0.5] ]

    # store the x-limits in the plots of each variable

    #            #[   T,   Q,    U,    V,   H]
    # RMSExmins = [ 0.0, 0.0, 1.25, 1.25, 0.0]
    # RMSExmaxs = [ 4.0, 4.0, 3.25, 3.25, 4.0]

    # BIASxmins = [-2.0,-2.0, -1.7, -1.7,-2.0]
    # BIASxmaxs = [ 2.0, 2.0,  1.7,  1.7, 2.0]

    MEANxmins = [-0.03,-0.025]
    MEANxmaxs = [ 0.09, 0.025]

    # MEANxmins = [298]
    # MEANxmaxs = [311]

    thickness = 1.8 # set line thicknesses in plot


    # calculate the index in "alts" corresponding to this altitude
    maxAlti = int( (maxAlt / 10) + 1 )


    # CREATE AN ARRAY WHICH LISTS THE CORRESPONDING INDICES OF THE XSTRS CHOSEN
    # IN THE ORIGINAL MASTER XSTR LIST (this is here because np.find doesn't work)
    # create an empty array, must be filled with -1 to be an integer
    corr_xstri = np.full(np.size(ChosenXstrs), -1)

    for chosexstri in range(0, np.size(ChosenXstrs)):
        for allxstri in range(0, np.size(xStrs)):
            if (ChosenXstrs[chosexstri] == xStrs[allxstri]):
                corr_xstri[chosexstri] = allxstri

    print(corr_xstri)

    for Var in Vars + TurbVars:

        # retrieve the index of the chosen variable within its respective array (for labelling purposes)
        if (Var in Vars):
            for i in range(0, np.size(Vars)):
                if (Var == Vars[i]):
                    vari = i

        if (Var in TurbVars):
            for i in range(0, np.size(TurbVars)):
                if (Var == TurbVars[i]):
                    turbvari = i


        for chosexstri in range(0, np.size(ChosenXstrs)):
            for allxstri in range(0, np.size(xStrs)):
                if (ChosenXstrs[chosexstri] == xStrs[allxstri]):
                    corr_xstri[chosexstri] = allxstri

    #     set the x-limits on the plots

        if (Var in TurbVars):
            RMSExmin = RMSExmins[turbvari]
            RMSExmax = RMSExmaxs[turbvari]
            BIASxmin = BIASxmins[turbvari]
            BIASxmax = BIASxmaxs[turbvari]
            MEANxmin = MEANxmins[turbvari]
            MEANxmax = MEANxmaxs[turbvari]

        if (Var in Vars):
            RMSExmin = RMSExmins[vari]
            RMSExmax = RMSExmaxs[vari]
            BIASxmin = BIASxmins[vari]
            BIASxmax = BIASxmaxs[vari]
            MEANxmin = MEANxmins[vari]
            MEANxmax = MEANxmaxs[vari]



        # count the number of nudged runs included by the user
        NumOfNudgedRuns = 0
        if ('x501' in ChosenXstrs):
            NumOfNudgedRuns = NumOfNudgedRuns + 1
        if ('x511' in ChosenXstrs):
            NumOfNudgedRuns = NumOfNudgedRuns + 1
        if ('x521' in ChosenXstrs):
            NumOfNudgedRuns = NumOfNudgedRuns + 1

        # if the variable is a turbulence variable, don't include the nudged runs
        if (Var in Vars):
            LastXstr = np.size(ChosenXstrs)
        elif(Var in TurbVars):
            LastXstr = np.size(ChosenXstrs) - NumOfNudgedRuns


        for lead in [1]: #range(0,3):
            # PLOTTING SECTION

            # select only the altitudes beneath the chosen maximum
    #         AltsToPlot = alts[0:500] * 0.001

            plt.figure()

            # retrieve the means from model output for each run
            local_xstri = -1  # set starting local loop index
            for xStr in ChosenXstrs[0:LastXstr] :
                local_xstri = local_xstri + 1
                # retrieve the index of the chosen xStr in the master xStr list for plotting purposes
                xstri = corr_xstri[local_xstri]

                print(xstri)

                # retrievet the RMSE storage array for this variable
                exec("MeansToPlot = " + Var + "_MeansArray[xstri,lead,0:maxAlti]")
                # select only the altitudes beneath the chosen macumum
                AltsToPlot = alts[0:maxAlti]

                # PLOT THE ACTUAL LINE
#                 plt.plot(MeansToPlot, AltsToPlot*0.001, color = PlotxStrColours[xstri], \
#                          linestyle='--', linewidth=thickness)
# CMZ add to have dots for x001 for consistency
                if (xStr == 'x001'):
                    plt.plot(MeansToPlot, AltsToPlot*0.001, color = PlotxStrColours[xstri], \
                         linestyle=':', linewidth=thickness)
                else:
                    plt.plot(MeansToPlot, AltsToPlot*0.001, color = PlotxStrColours[xstri], \
                         linestyle='--', linewidth=thickness)

            # Check if Var exists in dictionary
            if Var in cm1_data_dict:
                # Plot data for each file
                for i, datatmp in enumerate(cm1_data_dict[Var]):
                    plt.plot(datatmp[cm1_start_index::], cm1z[i][cm1_start_index::], color=cm1colors[i % len(cm1colors)], linewidth=1.5, label='CM1')
            else:
                print(f"CM1 Variable '{Var}' not recognized.")

            # plot accessories
            plt.grid('on', linewidth=0.25)

    #         plt.legend(ChosenXstrs[0:LastXstr])

            # label with the corresponding variable name whether its a state or a turbulence variable
            if  (Var in Vars):
    #             plt.title(str(lead) + '-Day Lead Vertical Mean Profiles for ' + VarLongNames[vari])
                plt.xlabel('Mean ' + VarLongNames[vari] + ' [' + VarUnitses[vari] + ']',fontsize=plot_fontsize )
            elif (Var in TurbVars):
    #             plt.title(str(lead) + '-Day Lead Vertical Mean Profiles for ' + TurbVarLongNames[turbvari])
                plt.xlabel('Mean ' + TurbVarLongNames[turbvari]  + ' [' + TurbVarUnitses[turbvari] + ']' ,fontsize=plot_fontsize)

            plt.ylabel('Altitude [km]',fontsize=plot_fontsize)

            plt.xlim([MEANxmin,MEANxmax])
            plt.ylim([0,maxAlt*0.001])


            # include a thin black y-axis line
            plt.plot([0,0],[0,maxAlt],'k', linewidth=0.5)

            SaveFolder = VOLNAME+'/ThesisPlots/'


            if not os.path.exists(SaveFolder):
                os.makedirs(SaveFolder)

            savefigname=SaveFolder + Var + '_MeanProfiles' + str(lead) + 'DayLead' + \
                        ChosenXstrs[0] + ChosenXstrs[LastXstr-1] + '_2500mTopSansTitleSanLegend.'+FILEOUTTYPE
            plt.savefig( savefigname, \
                        facecolor='w',dpi=300)
            cmz_writing_fig(savefigname,5)
            plt.close()
    #         plt.savefig( SaveFolder + Var + '_MeanProfiles' + str(lead) + 'DayLead' + \
    #                     ChosenXstrs[0] + ChosenXstrs[LastXstr-1] + '_2500mTop.'+FILEOUTTYPE, facecolor='w',dpi=300)
    #         plt.close()


print_break()



# THIS BLOCK WRITTEN 2023-04-12W TAKES ERRORS BASED IN ALTITUDE

# THIS BLOCK RECREATES VERSIONS OF SAVAZZI FIGURE 11: DIURNAL VARIABLILITY OF FORECAST ERRORS

NumHourBlocks = 8

Alts = np.linspace(0,31000,3101)

BotAlt  =    0  # [m]
TopAlt  = 5000  # [m]

BotAlti = int(BotAlt/10)
TopAlti = int(TopAlt/10) + 1

NumAlts = TopAlti - BotAlti

# CHOOSE WHETHER YOU WANT THE PLOTS IN LOCAL TIME 'LT' or UNIVERSAL TIME 'UTC'
# note that local time will actually plot data from 23:00 - 02:00 (27 hours) to fill the 24 hour plot
TimeMode = 'LT'


# Meshes need to inlcude 9 slots for local time because the data are stored in UTC blocks
if (TimeMode == 'LT'):

    # create meshes for time and altitude level for pcolormesh [4 hours behind UTC 3-hour blocks]
    TimeMesh = np.repeat([[0.5, 3.5, 6.5, 9.5, 12.5, 15.5, 18.5, 21.5, 24.5]], NumAlts, axis=0)
    TimeMesh = TimeMesh.reshape(NumAlts,NumHourBlocks+1)
    # NOTE THAT THERE IS ONE MORE TIME SLOT IN LOCAL TIME SINCE IT LOOPS AROUND MIDNIGHT

    AltMesh = np.repeat( [Alts[BotAlti:TopAlti]], 9, axis=1)
    AltMesh = AltMesh.reshape(NumAlts,NumHourBlocks+1)

elif (TimeMode == 'UTC'):

    # create meshes for time and altitude level for pcolormesh
    TimeMesh = np.repeat([[1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5]], NumAlts, axis=0)
    TimeMesh = TimeMesh.reshape(NumAlts,NumHourBlocks)

    AltMesh = np.repeat( [Alts[BotAlti:TopAlti]], 8, axis=1)
    AltMesh = AltMesh.reshape(NumAlts,NumHourBlocks)

else:
    print("TimeMode needs to be 'LT' or 'UTC'")



# Variables and Corresponding units and Colour Bar Limits
Vars           = [   'T',    'Q',    'U',    'V', 'Hwind']
VarUnitses     = [   'K', 'g/kg',  'm/s',  'm/s', 'm/s'  ]
VarColorLimses = [[-2,2], [-2,2], [-2,2], [-2,2], [-2,2] ]

VarLongNames = ['Temperature', 'Water Vapor Mixing Ratio', 'U-Wind Velocity', 'V-Wind Velocity', \
                'Horizontal Wind Speed', 'Potential Temperature']


HourBlockStrings = ['0003', '0306', '0609', '0912', '1215', '1518', '1821', '2100']

HourBlockLength = 3
OverallStartHourStr = '00'
TopAlt = 4000


# LIST OF MODEL CONFIGURATIONS
xStrs = ['x001', 'x101', 'x201', 'x202' , 'x203', 'x204', 'x301', 'x302' , 'x303', 'x304']

# WHICH LEAD DAY FORECAST?
lead = 1




# LOOP OVER EACH MODEL CONFIG AND EACH VARIABLE
for xstri in range(0, np.size(xStrs)):
    xStr = xStrs[xstri]
    for vari in range(0, np.size(Vars)):
        VarStr       = Vars[vari]
        VarUnits     = VarUnitses[vari]
        VarColorLims = VarColorLimses[vari]
        VarLongName  = VarLongNames[vari]

        # create an empty array for storing domain means at each level and time interval
        if (TimeMode == 'LT'):
            ErrorArray2DLevTime     = np.full([NumAlts,NumHourBlocks+1], np.nan)
            ErrorArray2DLevTimex001 = np.full([NumAlts,NumHourBlocks+1], np.nan)
            ObsArray2DLevTime       = np.full([NumAlts,NumHourBlocks+1], np.nan)

        elif (TimeMode == 'UTC'):
            ErrorArray2DLevTime     = np.full([NumAlts,NumHourBlocks], np.nan)
            ErrorArray2DLevTimex001 = np.full([NumAlts,NumHourBlocks], np.nan)
            ObsArray2DLevTime       = np.full([NumAlts,NumHourBlocks], np.nan)



        # loop over each 3-hour block
        for hourblocki in range(0,NumHourBlocks):
            HourBlockStr = HourBlockStrings[hourblocki]


#             ErrorArray1DLev_name = VOLNAME+'/VARIABLES/' + str(HourBlockLength) + \
#                                       'HourBlocks/' + OverallStartHourStr + 'UTCstart/' + \
#                                       'Errors/AltBased/LeadDay' + \
#                                       str(lead) + '/' + xStr + '/' + VarStr + 'Error_' + xStr + 'LeadDay' + \
#                                       str(lead) + '_' + HourBlockStr + 'MeanProfile.npy'
            ErrorArray1DLev_name = VOLNAME+'/ThesisVariables/' + \
                                      VarStr + 'Error_' + xStr + 'LeadDay' + \
                                      str(lead) + '_' + HourBlockStr + 'MeanProfile.npy'
            ErrorArray1DLev = np.load(ErrorArray1DLev_name)
            cmz_print_filename(ErrorArray1DLev_name)

#             ObsArray1DLev_name = VOLNAME+'/VARIABLES/' + str(HourBlockLength) + \
#                                     'HourBlocks/' + OverallStartHourStr + 'UTCstart/' + \
#                                     'Observations/AltBased/' + VarStr + 'Value_Obs_' + HourBlockStr + \
#                                     'MeanProfile.npy'
            ObsArray1DLev_name = VOLNAME+'/ThesisVariables/' + \
                                    VarStr + 'Value_Obs_' + HourBlockStr + \
                                    'MeanProfile.npy'
            ObsArray1DLev = np.load(ObsArray1DLev_name)
            cmz_print_filename(ObsArray1DLev_name)

            # take that mean column and put it in the (levels x time) array

            # rearrange 3-hourly data for local time
            if (TimeMode == 'LT'):

                # the second to last 3-hour block in local time is the first 3-hour block in UTC
                if (hourblocki == 0):
                    ErrorArray2DLevTime[:,7] = ErrorArray1DLev[BotAlti:TopAlti]
                    ObsArray2DLevTime[:,7]   = ObsArray1DLev[BotAlti:TopAlti]
                # the first and last 3-hour blocks in local time are the second 3-hour block in UTC
                elif (hourblocki == 1):
                    ErrorArray2DLevTime[:,0] = ErrorArray1DLev[BotAlti:TopAlti]
                    ErrorArray2DLevTime[:,8] = ErrorArray1DLev[BotAlti:TopAlti]
                    ObsArray2DLevTime[:,0]   = ObsArray1DLev[BotAlti:TopAlti]
                    ObsArray2DLevTime[:,8]   = ObsArray1DLev[BotAlti:TopAlti]
                # all other 3-hour UTC blocks are 1 block earlier in local time
                else:
                    ErrorArray2DLevTime[:,hourblocki-1] = ErrorArray1DLev[BotAlti:TopAlti]
                    ObsArray2DLevTime[:,hourblocki-1]   = ObsArray1DLev[BotAlti:TopAlti]

            # don't rearrange for UTC!!!
            elif (TimeMode == 'UTC'):
                ErrorArray2DLevTime[:,hourblocki] = ErrorArray1DLev[BotAlti:TopAlti]
                ObsArray2DLevTime[:,hourblocki]   = ObsArray1DLev[BotAlti:TopAlti]


#             ErrorArray1DLevx001_name = VOLNAME+'/VARIABLES/' + str(HourBlockLength) + \
#                                           'HourBlocks/' + \
#                           OverallStartHourStr + 'UTCstart/' + 'Errors/AltBased/LeadDay' + \
#                           str(lead) + '/x001/' + VarStr + 'Error_x001LeadDay' + \
#                           str(lead) + '_' + HourBlockStr + 'MeanProfile.npy'
            ErrorArray1DLevx001_name =VOLNAME+'/ThesisVariables/' + \
                          VarStr + 'Error_x001LeadDay' + \
                          str(lead) + '_' + HourBlockStr + 'MeanProfile.npy'
            ErrorArray1DLevx001 = np.load(ErrorArray1DLevx001_name)
            cmz_print_filename(ErrorArray1DLevx001_name)


            # rearrange 3-hourly data for local time
            if (TimeMode == 'LT'):

                # the second to last 3-hour block in local time is the first 3-hour block in UTC
                if (hourblocki == 0):
                    ErrorArray2DLevTimex001[:,7] = ErrorArray1DLevx001[BotAlti:TopAlti]
                # the first and last 3-hour blocks in local time are the second 3-hour block in UTC
                elif (hourblocki == 1):
                    ErrorArray2DLevTimex001[:,0] = ErrorArray1DLevx001[BotAlti:TopAlti]
                    ErrorArray2DLevTimex001[:,8] = ErrorArray1DLevx001[BotAlti:TopAlti]
                # all other 3-hour UTC blocks are 1 block earlier in local time
                else:
                    ErrorArray2DLevTimex001[:,hourblocki-1] = ErrorArray1DLevx001[BotAlti:TopAlti]

            # don't rearrange for UTC!!!
            elif (TimeMode == 'UTC'):
                # take that mean column and put it in the (levels x time) array
                ErrorArray2DLevTimex001[:,hourblocki] = ErrorArray1DLevx001[BotAlti:TopAlti]



        # use continuous colour scale for non-bias plots
        ccmap = 'viridis'


        # ACTUAL PLOTS

        # SAVE EACH PLOT AS A FILE
        SaveFolder = VOLNAME+'/ThesisPlots/'

        if not os.path.exists(SaveFolder):
            os.makedirs(SaveFolder)

        SaveFile = VarStr + '_TimeHeightObs' + str(lead) + 'DayLead' + xStr + TimeMode + 'SansLegend.'+FILEOUTTYPE

#         plt.savefig( SaveFolder + SaveFile, facecolor='w', bbox_inches='tight', dpi=300)


        # use purple/green on the Hwind plots and blue/red on all other plots
        if (VarStr == 'Hwind'):
            ccmap = 'PRGn_r'
        else:
            ccmap = 'coolwarm_r'

        # BIASES
        fig = plt.figure(figsize=(5,2))
        ax = fig.add_subplot(1, 1, 1)
        plt.pcolormesh(TimeMesh, AltMesh*0.001, ErrorArray2DLevTime, shading = 'nearest', cmap=ccmap)
        plt.colorbar(label='Model Bias' + ' [' + VarUnits + ']')


        # limits and grids

        # Major ticks every 6 hours, minor ticks every 1
        major_xticks = np.arange(0, 25, 6)
        minor_xticks = np.arange(0, 25, 1)
        ax.set_xticks(major_xticks)
        ax.set_xticks(minor_xticks, minor=True)

        # Major ticks every 1 km, minor ticks every 500 m
        major_yticks = np.arange(0, 6,   1)
        minor_yticks = np.arange(0, 6, 0.5)
        ax.set_yticks(major_yticks)
        ax.set_yticks(minor_yticks, minor=True)

        # Make the major lines darker than the minor lines
#         ax.grid(which='minor', color='k', alpha=0.1)
        ax.grid(which='major', color='k', alpha=0.2)

        # viewing limits
        plt.xlim([0,24])
        plt.ylim([0,5])

        plt.clim(VarColorLims)




        # labels

        # label the plot in the top left corner a), b), or c) for figures in LaTeX
        FigLabels = ['a', 'b', 'c', \
                     'd', 'e', 'f', \
                     'g', 'h', 'i', \
                     'j', 'k', 'l', \
                     'm', 'n', 'o', \
                     'p', 'q', 'r', \
                     's', 't', 'u', \
                     'v', 'w', 'x', \
                     'y', 'z', 'aa', \
                     'bb', 'cc', 'dd', \
                       ]

        if   (VarStr == 'U'):
            PlotLabelText = FigLabels[0 + xstri*3] + ')'
        elif (VarStr == 'V'):
            PlotLabelText = FigLabels[1 + xstri*3] + ')'
        elif (VarStr == 'Hwind'):
            PlotLabelText = FigLabels[2 + xstri*3] + ')'
        else:
            PlotLabelText = ''

        # place the label at 00:30 and 3.8 km
        Xposition = 0.5
        Yposition = 3.8
        plt.text(Xposition,Yposition, PlotLabelText ,fontsize=24,weight='bold')


        # labels
        plt.xlabel('Time of Day [' + TimeMode + ' Hour]')
        if (TimeMode == 'LT'):
             plt.xlabel('Time of Day [Local Time Hour]')
        plt.ylabel('Altitude [km]')
        plt.title(VarLongName + ' Bias in ' + xStrs[xstri])


        # SAVE EACH PLOT AS A FILE
        SaveFolder = VOLNAME+'/ThesisPlots/'


        if not os.path.exists(SaveFolder):
            os.makedirs(SaveFolder)

        SaveFile = VarStr + '_TimeHeightBias' + str(lead) + 'DayLead' + xStr + TimeMode + '.'+FILEOUTTYPE

        plt.savefig( SaveFolder + SaveFile, facecolor='w', bbox_inches='tight', dpi=300)
        cmz_writing_fig(SaveFolder + SaveFile,6)
        plt.close()

print_break()



# THIS BLOCK WRITTEN 2023-04-12W TAKES ERRORS BASED IN ALTITUDE

# THIS BLOCK RECREATES VERSIONS OF SAVAZZI FIGURE 7 ABC: DIURNAL VARIABLILITY OF WINDS


# width of the lines on the plot
thickness = 2


NumHourBlocks = 8

Alts = np.linspace(0,31000,3101)

BotAlt  =  200   # [m]
TopAlt  =  2000   # [m]

BotAlti = int(BotAlt/10)
TopAlti = int(TopAlt/10) + 1

NumAlts = TopAlti - BotAlti

# CHOOSE WHETHER YOU WANT THE PLOTS IN LOCAL TIME 'LT' or UNIVERSAL TIME 'UTC'
# note that local time will actually plot data from 23:00 - 02:00 (27 hours) to fill the 24 hour plot
TimeMode = 'LT'


# Meshes need to inlcude 9 slots for local time because the data are stored in UTC blocks

# ADD EXTRA HOURS SO THAT THE LINES LOOP
if (TimeMode == 'LT'):
    PlotTimes = [-2.5, 0.5, 3.5, 6.5, 9.5,  12.5, 15.5, 18.5, 21.5, 24.5]

elif (TimeMode == 'UTC'):
    PlotTimes = [-1.5, 1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5, 25.5]

else:
    print("TimeMode needs to be 'LT' or 'UTC'")


xStrs = ['x001', 'x101', 'x201', 'x202' , 'x203', 'x204', 'x301', 'x302' , 'x303', 'x304']


# colours associated with each of the model runs
PlotxStrColours = [ [0.9,0.0,0.0],
                    [0.0,0.5,0.0], \
                    [0.0,0.85,1.0], [0.0,0.6,1.0], [0.0,0.0,0.9], [0.4,0.0,0.5], \
                    [0.5,0.85,1.0], [0.5,0.6,1.0], [0.5,0.0,0.9], [0.4,0.5,0.5] ]



# Variables and Corresponding units and Colour Bar Limits
Vars           = [   'T',    'Q',      'U',     'V',   'Hwind']
VarUnitses     = [   'K', 'g/kg',    'm/s',   'm/s',   'm/s'  ]
VarYLimses     = [[-2,2], [-2,2], [-9.5,-6.5], [-3,-0], [7,10] ]

VarLongNames = ['Temperature', 'Water Vapor Mixing Ratio', 'U-Wind Velocity', 'V-Wind Velocity', \
                'Horizontal Wind Speed', 'Potential Temperature']



HourBlockStrings = ['0003', '0306', '0609', '0912', '1215', '1518', '1821', '2100']

HourBlockLength = 3
OverallStartHourStr = '00'
# TopAlt = 5000


# LOOP OVER EACH VARIABLE

for vari in range(2,5): # np.size(Vars)):
    VarStr       = Vars[vari]
    VarUnits     = VarUnitses[vari]
    VarYLims     = VarYLimses[vari]
    VarLongName  = VarLongNames[vari]


    # create an empty array for storing domain means at each level and time interval
    ErrorArray2DLevTime     = np.full([21,8], np.nan)
    ErrorArray2DLevTimex001 = np.full([21,8], np.nan)


#     # loop over each 3-hour block
#     for hourblocki in range(0,8):



    # FOR OBSERVATIONS

    # collect the mean of this variable for the specified range of altitudes
    # add 2 to the size of number of hour blocks to have the line connect to points on either side
    HourlyMeans = np.full(NumHourBlocks+2, np.nan)

    for hbi in range(0,NumHourBlocks):
        HourBlockStr = HourBlockStrings[hbi]

        HourBlockMeanArray_name = VOLNAME+'/ThesisVariables/' + \
                                    VarStr + 'Value_Obs_' + HourBlockStr + '.npy'
        HourBlockMeanArray = np.load(HourBlockMeanArray_name)
        cmz_print_filename(HourBlockMeanArray_name)


        HourlyMeans[hbi] = np.nanmean(HourBlockMeanArray[:,BotAlti:TopAlti])

    # rearrange for LOCAL TIME or add extra end hours for UTC for looping
        if (TimeMode == 'LT'):
            HourlyMeansLT = np.full( NumHourBlocks+2, np.nan)

            # the second to last 3-hour block in local time is the first 3-hour block in UTC
            HourlyMeansLT[8] = HourlyMeans[0]
            # the last 3-hour block in local time is the second 3-hour block in UTC
            HourlyMeansLT[9] = HourlyMeans[1]
            # all other 3-hour UTC blocks are the same block in local time
            for i in range(0,8):
                HourlyMeansLT[i] = HourlyMeans[i]

            HourlyMeansObs = HourlyMeansLT[:]


        elif (TimeMode == 'UTC'):
            HourlyMeansUTC = np.full( np.size(HourlyMeans)+2, np.nan)

            # add the last block as the first block
            HourlyMeansUTC[0] = HourlyMeans[7]
            # add the first block as the last block
            HourlyMeansUTC[9] = HourlyMeans[0]
            # move all other blocks up one
            for i in range(0,8):
                HourlyMeansUTC[i+1] = HourlyMeans[i]

            HourlyMeansObs = HourlyMeansUTC[:]



    # FOR MODEL RUNS

    # collect the mean of this variable for the specified range of altitudes
    # add 2 to the size of number of hour blocks to have the line connect to points on either side

    lead = 1

    for xStr in xStrs:

        HourlyMeans = np.full(NumHourBlocks+2, np.nan)

        for hbi in range(0,NumHourBlocks):
            HourBlockStr = HourBlockStrings[hbi]

            HourBlockMeanArray_name = VOLNAME+'/ThesisVariables/' + \
                                         VarStr + 'Value_' + \
                                         xStr + 'LeadDay' + str(lead) + '_' + HourBlockStr + '.npy'
            HourBlockMeanArray = np.load(HourBlockMeanArray_name)
            cmz_print_filename(HourBlockMeanArray_name)

            HourlyMeans[hbi] = np.nanmean(HourBlockMeanArray[:,BotAlti:TopAlti])

        # rearrange for LOCAL TIME or add extra end hours for UTC for looping
            if (TimeMode == 'LT'):
                HourlyMeansLT = np.full( NumHourBlocks+2, np.nan)

                # the second to last 3-hour block in local time is the first 3-hour block in UTC
                HourlyMeansLT[8] = HourlyMeans[0]
                # the last 3-hour block in local time is the second 3-hour block in UTC
                HourlyMeansLT[9] = HourlyMeans[1]
                # all other 3-hour UTC blocks are the same block in local time
                for i in range(0,8):
                    HourlyMeansLT[i] = HourlyMeans[i]

                exec("HourlyMeans" + xStr + " = HourlyMeansLT[:]")


            elif (TimeMode == 'UTC'):
                HourlyMeansUTC = np.full( np.size(HourlyMeans)+2, np.nan)

                # add the last block as the first block
                HourlyMeansUTC[0] = HourlyMeans[7]
                # add the first block as the last block
                HourlyMeansUTC[9] = HourlyMeans[0]
                # move all other blocks up one
                for i in range(0,8):
                    HourlyMeansUTC[i+1] = HourlyMeans[i]

                exec("HourlyMeans" + xStr + " = HourlyMeansUTC[:]")

    # ACTUAL PLOTS
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(1, 1, 1)

    plt.plot(PlotTimes, HourlyMeansObs,  color = [0.0,0.0,0.0], linewidth=2.0, label = 'Observations')

    for xstri in range(0,np.size(xStrs)):
        xStr = xStrs[xstri]

        if (xStr == 'x001'): # add dotted line for x001 and dashed line for prognostic runs
            exec("plt.plot(PlotTimes, HourlyMeans" + xStr + ", color = " + str(PlotxStrColours[xstri])  + \
                 ", linestyle=':', linewidth = " + str(thickness) + ", label=xStr)" )
        else:
            exec("plt.plot(PlotTimes, HourlyMeans" + xStr + ", color = " + str(PlotxStrColours[xstri])  + \
                 ", linestyle='--', linewidth = " + str(thickness) + ", label=xStr)" )

    #             ax.legend(bbox_to_anchor=(1.01, 1.02))

    # limits and grids

    # Major ticks every 6 hours, minor ticks every 1
    major_xticks = np.arange(0, 25, 6)
    minor_xticks = np.arange(0, 25, 1)
    ax.set_xticks(major_xticks)
    ax.set_xticks(minor_xticks, minor=True)

    # Major ticks every 1 m/s, minor ticks every 0.25 (over sufficent range from -20 to 20 [m/s] for all)
    major_yticks = np.arange(-20, 20,    1)
    minor_yticks = np.arange(-20, 20, 0.25)
    ax.set_yticks(major_yticks)
    ax.set_yticks(minor_yticks, minor=True)

    # Make the major lines darker than the minor lines
    ax.grid(which='minor', alpha=0.1)
    ax.grid(which='major', alpha=0.5)

    # viewing limits
    plt.xlim([0,24])
    plt.ylim(VarYLims)

    # labels
    plt.xlabel('Hour of Day [' + TimeMode + ']', fontsize=18)
    if (TimeMode == 'LT'):
            plt.xlabel('Hour of Day [Local Time]')

    # use smaller font size for long name on y-axis label
    if (VarLongName == 'Horizontal Wind Speed'):
        plt.ylabel(VarLongName + ' [' + VarUnits + ']', fontsize=11)
    else:
        plt.ylabel(VarLongName + ' [' + VarUnits + ']', fontsize=14)

    plt.title( 'Diurnal Cycle of ' + str(BotAlt) + '-' + str(TopAlt) + ' m Mean ' + VarLongName)

    # label the plot in the top left corner a), b), or c) for figures in LaTeX
    if   (VarStr == 'U'):
        PlotLabelText = 'a)'
    elif (VarStr == 'V'):
        PlotLabelText = 'b)'
    elif (VarStr == 'Hwind'):
        PlotLabelText = 'c)'

    # place the label at 00:30 and 15% the way down from the top
    Xposition = 0.5
    Yposition = np.max(VarYLims) - ( (np.max(VarYLims) - np.min(VarYLims)) * 0.15)
    plt.text(Xposition,Yposition, PlotLabelText ,fontsize=24,weight='bold')

    #     SAVE EACH PLOT AS A FILE
    SaveFolder = VOLNAME+'/ThesisPlots/'

    if not os.path.exists(SaveFolder):
        os.makedirs(SaveFolder)

    SaveFile = VarStr + '_TimePlot' + TimeMode + str(BotAlt) + 'mTO' + str(TopAlt) + 'm' + xStrs[0] + \
    xStrs[-1] + 'SansLegend.'+FILEOUTTYPE

    plt.savefig( SaveFolder + SaveFile, facecolor='w', bbox_inches='tight', dpi=300)
    cmz_writing_fig(SaveFolder + SaveFile,7)
    plt.close()


print_break()



# THIS BLOCK CREATES PLOTS OF THE 1D LEVEL MEANS FOR EACH VARIABLE
#[U SECTION]

# altitudes in km
PlotAlts = np.arange(0,30000,10) * 0.001

# names of variables stored in .npy files
LoadVars = ['upwp_bt' , 'upwp_ma' , 'upwp_ta' , 'upwp_tp' , 'upwp_ac' , 'upwp_bp',
            'upwp_pr1', 'upwp_pr2', 'upwp_pr3', 'upwp_pr4', 'upwp_dp1', 'upwp_cl', 'upwp_mfl']

# name of variables to plot in figures
PlotVars =     ['MeanAdvection', 'TurbulentAdvection',  \
                'VerticalWindTurbulentProduction', 'VerticalGradientTurbulentProduction', \
                'BuoyantProduction', 'ReturnToIsotropy', \
                'Residual', \
                'TimeTendency']

# what to label them in the legend
PlotVarLabels = ['(1) Mean Advection'                       ,'(2) Turbulent Advection', \
                 '(3) Turbulent Production by Vertical Wind','(4) Turbulent Production by Vertical Gradient', \
                 '(5) Buoyant Production'                   ,'(6) Return to Isotropy',  \
                 '(7) Residual', \
                 'Time Tendency x 10 (left hand side)']

# colours associated with each momentum component variable (variables plotted in figures)
PlotVarColours = [ [0.9,0.0,0.0], [0.9,0.0,0.0], \
                   [0.5,0.2,0.0], [0.5,0.2,0.0], \
                   [0.2,0.4,0.7], [0.5,0.0,0.8], \
                   [0.5,0.5,0.5],
                   [0.0,0.0,0.0] ]

# line styles associated with each momentum component variable (variables plotted in figures)
PlotVarLines   = [ '-', ':', \
                   '-', ':', \
                   '-', ':', \
                   ':', \
                   '--']


# set the axis limits on the plots
MaxPlotAlt = 2.5        # [km] altitude at the top of the plots
MinPlotUPWP = -0.00026  # [m^2/s^3] minimum momentum flux tendency on the plots
MaxPlotUPWP =  0.00026  # [m^2/s^3] maximum momentum flux tendency on the plots


# model configurations to plot
xStrs = ['x101', 'x204']
# xStrs = ['x101', 'x201', 'x202', 'x203', 'x204']



# LOOP OVER EACH MODEL CONFIGURATION
for xstri in range(0, np.size(xStrs)):
    xStr = xStrs[xstri]

    # CALCULATE VARIABLES TO PLOT
    for loadvari in range(0,np.size(LoadVars)):
        LoadVar = LoadVars[loadvari]

        VarArray_name = VOLNAME+'/ThesisVariables/' + \
                           'SoundingGrand_' + LoadVar + xStr + 'Valt.npy'
        VarArray = np.load(VarArray_name)
        cmz_print_filename(VarArray_name)

        exec(LoadVar + "_means = np.nanmean(VarArray,axis=0)")

    MeanAdvection_Means                       = upwp_ma_means
    TurbulentAdvection_Means                  = upwp_ta_means
    VerticalWindTurbulentProduction_Means     = upwp_tp_means  + upwp_pr4_means
    VerticalGradientTurbulentProduction_Means = upwp_ac_means  + upwp_pr2_means
    BuoyantProduction_Means                   = upwp_bp_means  + upwp_pr3_means
    ReturnToIsotropy_Means                    = upwp_pr1_means
    Residual_Means                            = upwp_dp1_means + upwp_cl_means  + upwp_mfl_means
    TimeTendency_Means                        = upwp_bt_means * 10


    # PLOT FIGURE
    ax = plt.figure()

    # add a black y-axis line
    plt.plot([0,0],[0,MaxPlotAlt],'k',linewidth=0.5)

    for plotvari in range(0,np.size(PlotVars)):
        PlotVar = PlotVars[plotvari]

        # plot actual lines (exclude the lowest level which is below 0 m)
        exec("plt.plot(" + PlotVar + "_Means[0:250], PlotAlts[0:250], color = PlotVarColours[plotvari]," + \
                       "linestyle = PlotVarLines[plotvari], linewidth = 2, label=PlotVarLabels[plotvari])")

    plt.grid()

    if xStr == "x204":
        ax.legend(bbox_to_anchor=(1.52, 0.9))
    #    plt.legend()

    plt.title("Domain and Time Mean u'w' Components for " + xStr)

    plt.xlabel(" u'w' Budget Terms [m\u00b2/s\u00b3]",fontsize=plot_fontsize)
    plt.ylabel('Altitude [km]',fontsize=plot_fontsize)


    plt.ylim([ 0, MaxPlotAlt])
    plt.xlim([ MinPlotUPWP, MaxPlotUPWP])

    # add a black y-axis line
    plt.plot([0,0],[0,MaxPlotAlt],'k',linewidth=0.5)



    SaveFolder = VOLNAME+'/ThesisPlots/'

    if not os.path.exists(SaveFolder):
        os.makedirs(SaveFolder)

    if xStr == "x204":
        SaveFile = xStr + 'UPWPComponentProfilesValtx10RightLegend.'+FILEOUTTYPE
    else:
        SaveFile = xStr + 'UPWPComponentProfilesValtx10SansLegend.'+FILEOUTTYPE

    plt.savefig( SaveFolder + SaveFile, facecolor='w', bbox_inches='tight', dpi=300)
    cmz_writing_fig(SaveFolder + SaveFile,8)
    plt.close()


print_break()



# THIS BLOCK CREATES PLOTS OF THE 1D LEVEL MEANS FOR EACH VARIABLE
#[V SECTION]

# altitudes in km
PlotAlts = np.arange(0,30000,10) * 0.001

# names of variables stored in .npy files
LoadVars = ['vpwp_bt' , 'vpwp_ma' , 'vpwp_ta' , 'vpwp_tp' , 'vpwp_ac' , 'vpwp_bp',
            'vpwp_pr1', 'vpwp_pr2', 'vpwp_pr3', 'vpwp_pr4', 'vpwp_dp1', 'vpwp_cl', 'vpwp_mfl']

# name of variables to plot in figures
PlotVars =     ['MeanAdvection', 'TurbulentAdvection',  \
                'VerticalWindTurbulentProduction', 'VerticalGradientTurbulentProduction', \
                'BuoyantProduction', 'ReturnToIsotropy', \
                'Residual', \
                'TimeTendency']

# what to label them in the legend
PlotVarLabels = ['(1) Mean Advection'                       ,'(2) Turbulent Advection', \
                 '(3) Turbulent Production by Vertical Wind','(4) Turbulent Production by Vertical Gradient', \
                 '(5) Buoyant Production'                   ,'(6) Return to Isotropy',  \
                 '(7) Residual', \
                 'Time Tendency x 10 (left hand side)']

# colours associated with each momentum component variable (variables plotted in figures)
PlotVarColours = [ [0.9,0.0,0.0], [0.9,0.0,0.0], \
                   [0.5,0.2,0.0], [0.5,0.2,0.0], \
                   [0.2,0.4,0.7], [0.5,0.0,0.8], \
                   [0.5,0.5,0.5],
                   [0.0,0.0,0.0] ]

# line styles associated with each momentum component variable (variables plotted in figures)
PlotVarLines   = [ '-', ':', \
                   '-', ':', \
                   '-', ':', \
                   ':', \
                   '--']


# set the axis limits on the plots
MaxPlotAlt = 2.5       # [km] altitude at the top of the plots
MinPlotUPWP = -0.00026  # [m^2/s^3] minimum momentum flux tendency on the plots
MaxPlotUPWP =  0.00026  # [m^2/s^3] maximum momentum flux tendency on the plots


# model configurations to plot
xStrs = ['x101', 'x204']
# xStrs = ['x101', 'x201', 'x202', 'x203', 'x204']



# LOOP OVER EACH MODEL CONFIGURATION
for xstri in range(0, np.size(xStrs)):
    xStr = xStrs[xstri]

    # CALCULATE VARIABLES TO PLOT
    for loadvari in range(0,np.size(LoadVars)):
        LoadVar = LoadVars[loadvari]

        VarArray_name = VOLNAME+'/ThesisVariables/' + \
                           'SoundingGrand_' + LoadVar + xStr + 'Valt.npy'
        VarArray = np.load(VarArray_name)
        cmz_print_filename(VarArray_name)

        exec(LoadVar + "_means = np.nanmean(VarArray,axis=0)")

    MeanAdvection_Means                       = vpwp_ma_means
    TurbulentAdvection_Means                  = vpwp_ta_means
    VerticalWindTurbulentProduction_Means     = vpwp_tp_means  + vpwp_pr4_means
    VerticalGradientTurbulentProduction_Means = vpwp_ac_means  + vpwp_pr2_means
    BuoyantProduction_Means                   = vpwp_bp_means  + vpwp_pr3_means
    ReturnToIsotropy_Means                    = vpwp_pr1_means
    Residual_Means                            = vpwp_dp1_means + vpwp_cl_means  + vpwp_mfl_means
    TimeTendency_Means                        = vpwp_bt_means * 10


    # PLOT FIGURE
    ax = plt.figure()

    # add a black y-axis line
    plt.plot([0,0],[0,MaxPlotAlt],'k',linewidth=0.5)

    for plotvari in range(0,np.size(PlotVars)):
        PlotVar = PlotVars[plotvari]

        # plot actual lines (exclude the lowest level which is below 0 m)
        exec("plt.plot(" + PlotVar + "_Means[0:250], PlotAlts[0:250], color = PlotVarColours[plotvari]," + \
                       "linestyle = PlotVarLines[plotvari], linewidth = 2)")

    plt.grid()

    if xStr == "x204":
        ax.legend(bbox_to_anchor=(1.52, 0.9))
    #    plt.legend()

    plt.title("Domain and Time Mean v'w' Components for " + xStr)

    plt.xlabel(" v'w' Budget Terms [m\u00b2/s\u00b3]",fontsize=plot_fontsize)
    plt.ylabel('Altitude [km]',fontsize=plot_fontsize)


    plt.ylim([ 0, MaxPlotAlt])
    plt.xlim([ MinPlotUPWP, MaxPlotUPWP])

    # add a black y-axis line
    plt.plot([0,0],[0,MaxPlotAlt],'k',linewidth=0.5)


    SaveFolder = VOLNAME+'/ThesisPlots/'

    if not os.path.exists(SaveFolder):
        os.makedirs(SaveFolder)

    if xStr == "x204":
        SaveFile = xStr + 'VPWPComponentProfilesValtx10RightLegend.'+FILEOUTTYPE
    else:
        SaveFile = xStr + 'VPWPComponentProfilesValtx10SansLegend.'+FILEOUTTYPE

    plt.savefig( SaveFolder + SaveFile, facecolor='w', bbox_inches='tight', dpi=300)
    cmz_writing_fig(SaveFolder + SaveFile,9)
    plt.close()


print_break()



# THIS BLOCK WRITTEN 2023-03-26S TAKES ERRORS BASED IN ALTITUDE

# range of altitudes you want to take mean biases and RMSEs over
MinAlt =  200 # [m]
MaxAlt = 2000 # [m]
# find the corresponding indices of these altitude limits in the sounding data
MinAlti = int(MinAlt*0.1) -1
MaxAlti = int(MaxAlt*0.1) -1

# we are only lookin at 1-day lead errors here
lead = 1

# lists of the variables and model configurations
OriginalVarStrings = ['ta', 'q', 'u', 'v', 'theta','Hwind']
xStrs = ['x001', 'x101', 'x201', 'x202' , 'x301', 'x302']

for vari in range(0,np.size(OriginalVarStrings )):
    Var = OriginalVarStrings [vari]
    for xstri in range(0, np.size(xStrs)):
        xStr = xStrs[xstri]

        Root = VOLNAME+'/ThesisVariables/'
        Folder = ''
        BiasFile = Var + 'Errors_' + xStr + 'LeadDay' + str(lead) + 'MeanProfile.npy'
        RMSEFile = Var + 'Errors_' + xStr + 'LeadDay' + str(lead) + 'RMSEProfile.npy'

        BiasProfile = np.load(Root+Folder+BiasFile)
        RMSEProfile = np.load(Root+Folder+RMSEFile)
        cmz_print_filename(Root+Folder+BiasFile)
        cmz_print_filename(Root+Folder+RMSEFile)

        MeanBiasInRange = np.mean(BiasProfile[MinAlti:MaxAlti])
        MeanRMSEInRange = np.mean(RMSEProfile[MinAlti:MaxAlti])


print_break()



# THIS BLOCK CREATES STOPLIGHT DIAGRAMS FOR RMSE AND BIAS VALUES RELATIVE TO THE X001 RUN


# set which lead day you want to retrieve error stats for
lead = 1

# model runs stored in the error stat files
StoredXstrs = ['x001','x101','x201', 'x202' , 'x203', 'x204', 'x301', 'x302', 'x303', 'x304']

# model runs to include on the stoplight diagrams (AND CORRESPONDING INDEX IN ALL STORED XSTRS)
TableXstrs =  ['x001','x101','x201', 'x202' , 'x203', 'x204', 'x301', 'x302', 'x303', 'x304']
# CMZ, had to change these indices

TableXstris = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# variables to include on the stoplight diagrams
# TableVars = ['T', 'Q', 'U', 'V', 'Hwind']    # for old model level based version
TableVars = ['u', 'v', 'Hwind', 'ta', 'theta', 'q']

# create empty arrays for RMSEs and Biases of each model run and each variable
# and another set of arrays for how much each error value differs from the default run (x001)
# and a third set of arrays for what fraction of the default error this error difference is
RMSEarray = np.full( [np.size(TableVars), np.size(TableXstrs)] , np.nan)
BIASarray = np.full( [np.size(TableVars), np.size(TableXstrs)] , np.nan)
RMSEarrayDiffs = np.full( [np.size(TableVars), np.size(TableXstrs)] , np.nan)
BIASarrayDiffs = np.full( [np.size(TableVars), np.size(TableXstrs)] , np.nan)
RMSEarrayDiffRatios = np.full( [np.size(TableVars), np.size(TableXstrs)] , np.nan)
BIASarrayDiffRatios = np.full( [np.size(TableVars), np.size(TableXstrs)] , np.nan)


for vari in range(0, np.size(TableVars)):
    Var = TableVars[vari]

    xstri = 0                      # index for those model runs included
    for tablexstri in TableXstris: # index of all model runs in the stats file
        xStr = StoredXstrs[tablexstri]

        # NOTICE: REVERSING THE ORDER OF THE VARIABLES TO MATCH WITH PREVIOUS PLOTS!!!!
        # NOTICE: REVERSING THE ORDER OF THE VARIABLES TO MATCH WITH PREVIOUS PLOTS!!!!
        # NOTICE: REVERSING THE ORDER OF THE VARIABLES TO MATCH WITH PREVIOUS PLOTS!!!!
        Root = VOLNAME+'/ThesisVariables/'
        Folder = ''
        BiasFile = Var + 'Errors_' + xStr + 'LeadDay' + str(lead) + 'MeanProfile.npy'
        RMSEFile = Var + 'Errors_' + xStr + 'LeadDay' + str(lead) + 'RMSEProfile.npy'

        BiasProfile = np.load(Root+Folder+BiasFile)
        RMSEProfile = np.load(Root+Folder+RMSEFile)
        cmz_print_filename(Root+Folder+BiasFile)
        cmz_print_filename(Root+Folder+RMSEFile)

        RMSEarray[len(TableVars)-1-vari,xstri] = np.mean(RMSEProfile[MinAlti:MaxAlti])
        BIASarray[len(TableVars)-1-vari,xstri] = np.mean(BiasProfile[MinAlti:MaxAlti])

        xstri = xstri + 1


# calculate arrays for how much each error value differs from the default run (x001)
# and what fraction of the original value this difference is
for vari in range(0, len(TableVars)):
    for xstri in range(np.size(TableXstrs)):

        RMSEarrayDiffs[vari,xstri] = RMSEarray[vari, xstri] - RMSEarray[vari,0]
        BIASarrayDiffs[vari,xstri] = abs(BIASarray[vari, xstri]) - abs(BIASarray[vari,0])

        RMSEarrayDiffRatios[vari,xstri] = RMSEarrayDiffs[vari, xstri] / RMSEarray[vari,0]
        BIASarrayDiffRatios[vari,xstri] = BIASarrayDiffs[vari, xstri] / abs(BIASarray[vari,0])



# CREATE MESH ARRAYS FOR PLOTTING IN PCOLORMESH
# retrieve dimensions of the error stats
xDim = np.shape(RMSEarrayDiffs)[1]
yDim = np.shape(RMSEarrayDiffs)[0]

XmeshArray = np.linspace(0, xDim-1, xDim)
YmeshArray = np.linspace(0, yDim-1, yDim)

Xmesh = np.repeat([XmeshArray], yDim, axis=0)
Ymesh = np.repeat(  YmeshArray, xDim)

Xmesh = Xmesh.reshape(yDim, xDim)
Ymesh = Ymesh.reshape(yDim, xDim)


print_break()



# THIS BLOCK CREATES THE ACTUAL STOPLIGHT DIAGRAM PLOT



#RMSE SECTION
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

im = ax.pcolormesh(Xmesh, Ymesh, RMSEarrayDiffRatios, shading = 'nearest', \
                   cmap=plt.cm.get_cmap('seismic', 31), vmin=-0.15, vmax=0.15)


plt.title('CAM RMSEs Between ' + str(MinAlt) + ' m and ' + str(MaxAlt) + ' m Altitude' + '\n', fontsize=18)

ax.grid(color='black', linewidth=5, which='minor')

# set the limits to be just around the table
plt.xlim([-0.5, xDim-0.5])
plt.ylim([-0.5, yDim-0.5])



# plot lines to create a grid around each cell
# (work around for ax.grid(which='minor') not working)
for xi in range(0,yDim-1):
    ax.plot([0-0.5,xDim+0.5], [xi+0.5,xi+0.5], color='black', linewidth=2)

for yi in range(0,xDim-1):
    ax.plot([yi+0.5,yi+0.5], [0-0.5,yDim+0.5], color='black', linewidth=2)

# special thick lines seperating default run and optimised runs
ax.plot([0.5,0.5], [-0.5, 5.5], color='black', linewidth=5)
ax.plot([1.5,1.5], [-0.5, 5.5], color='black', linewidth=5)
if xDim > 6:
  ax.plot([5.5,5.5], [-0.5, 5.5], color='black', linewidth=5)

# special thick lines for the outside boarders
ax.plot([-0.5, xDim-0.5], [-0.5,-0.5], color='black', linewidth=5)
ax.plot([-0.5, xDim-0.5], [ yDim-0.5, yDim-0.5], color='black', linewidth=5)
ax.plot([-0.5,-0.5], [-0.5, yDim-0.5], color='black', linewidth=5)
ax.plot([ xDim-0.5, xDim-0.5], [-0.5, yDim-0.5], color='black', linewidth=5)


# long names of these variables IN REVERSE
TableVarLongNames = ['Mixing Ratio \n [g/kg]', 'Potential Temp \n [K]', 'Temperature \n [K]',
                     'Horizontal Wind Mag. \n [m/s]',   'V-wind \n [m/s]', 'U-wind \n [m/s]']

# main labels
# ax.set_title('State Variable Prediction Root Mean Squared Errors and' + '\n' + \
#              'Dependence on CAM Configuration', fontsize=24)
ax.set_xlabel('CAM Configuration', fontsize=22)
# ax.set_ylabel('Variable', fontsize=22)

# tick labels
ax.set_xticks(np.linspace(0, len(TableXstrs)-1, len(TableXstrs)) )
ax.set_yticks(np.linspace(0, len(TableVars) -1, len(TableVars)) )
ax.set_xticklabels(TableXstrs, fontsize=18) #, rotation=45)
ax.set_yticklabels(TableVarLongNames,  fontsize=15 , rotation=15)

# color bar and labels
cbar = fig.colorbar(im, pad=0.05, shrink=1, orientation = 'vertical' ,drawedges=True)
cbar.set_label('RMSE Relative to x001', fontsize=16)
cbar.set_ticks([-0.15, 0.0, 0.15])
cbar.set_ticklabels(['15% Decrease', 'Same', '15% Increase'])
cbar.ax.tick_params(labelsize=13)

if xDim > 6:
  thisFontSize=9
else:
  thisFontSize=18

# add numbers in each cell
for xstri in range(np.size(TableXstrs)):
    for vari in range(0, np.size(TableVars)):

            # retrieve cell value
            number = np.around(RMSEarray[vari,xstri], decimals=2)
            localcolor = np.around(RMSEarrayDiffRatios[vari,xstri], decimals=2)

            # place a black number if the cell is a light colour
            if (localcolor >= -0.05) and (localcolor <= 0.05) :
                text = ax.text(xstri-0.28, vari-0.08, str(number), c='black', fontsize=thisFontSize)
            else:
                text = ax.text(xstri-0.28, vari-0.08, str(number), c='white', fontsize=thisFontSize)
                text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

# SAVE THE RMSE PLOT
SaveFolder = VOLNAME+'/ThesisPlots/'

if not os.path.exists(SaveFolder):
    os.makedirs(SaveFolder)
SaveFile = 'TQUVHwindStoplightDiagramRMSE' + str(MinAlt) + 'mTo' + str(MaxAlt) + 'm.'+FILEOUTTYPE

plt.savefig( SaveFolder + SaveFile, facecolor='w', bbox_inches='tight', pad_inches=0.5, dpi=300)
cmz_writing_fig(SaveFolder + SaveFile,10)
plt.close()
# In[16]:


#BIAS SECTION
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

im = ax.pcolormesh(Xmesh, Ymesh, BIASarrayDiffRatios, shading = 'nearest', \
                   cmap=plt.cm.get_cmap('seismic', 17), vmin=-1.00, vmax=1.00)


plt.title('CAM Biases Between ' + str(MinAlt) + ' m and ' + str(MaxAlt) + ' m Altitude' + '\n', fontsize=18)

ax.grid(color='black', linewidth=5, which='minor')

# set the limits to be just around the table
plt.xlim([-0.5, xDim-0.5])
plt.ylim([-0.5, yDim-0.5])

# plot lines to create a grid around each cell
# (work around for ax.grid(which='minor') not working)
for xi in range(0,yDim-1):
    ax.plot([0-0.5,xDim+0.5], [xi+0.5,xi+0.5], color='black', linewidth=2)

for yi in range(0,xDim-1):
    ax.plot([yi+0.5,yi+0.5], [0-0.5,yDim+0.5], color='black', linewidth=2)

# special thick lines seperating default run and optimised runs
ax.plot([0.5,0.5], [-0.5, 5.5], color='black', linewidth=5)
ax.plot([1.5,1.5], [-0.5, 5.5], color='black', linewidth=5)
if xDim > 6:
  ax.plot([5.5,5.5], [-0.5, 5.5], color='black', linewidth=5)

# special thick lines for the outside boarders
ax.plot([-0.5, xDim-0.5], [-0.5,-0.5], color='black', linewidth=5)
ax.plot([-0.5, xDim-0.5], [ yDim-0.5, yDim-0.5], color='black', linewidth=5)
ax.plot([-0.5,-0.5], [-0.5, yDim-0.5], color='black', linewidth=5)
ax.plot([ xDim-0.5, xDim-0.5], [-0.5, yDim-0.5], color='black', linewidth=5)

# long names of these variables IN REVERSE
TableVarLongNames = ['Mixing Ratio \n [g/kg]', 'Potential Temp \n [K]', 'Temperature \n [K]',
                     'Horizontal Wind Mag. \n [m/s]',   'V-wind \n [m/s]', 'U-wind \n [m/s]']



# main labels
# ax.set_title('State Variable Prediction Mean Errors (Biases) and' + '\n' + \
#              'Dependence on CAM Configuration', fontsize=24)
ax.set_xlabel('CAM Configuration', fontsize=22)
# ax.set_ylabel('Variable', fontsize=22)

# tick labels
ax.set_xticks(np.linspace(0, len(TableXstrs)-1, len(TableXstrs)) )
ax.set_yticks(np.linspace(0, len(TableVars) -1, len(TableVars)) )
ax.set_xticklabels(TableXstrs, fontsize=18) #, rotation=45)
ax.set_yticklabels(TableVarLongNames,  fontsize=15 , rotation=15)

# color bar and labels
cbar = fig.colorbar(im, pad=0.05, shrink=1, orientation = 'vertical' ,drawedges=True)
cbar.set_label('Absolute Bias Relative to x001', fontsize=16)
cbar.set_ticks([-1.00, 0.0, 1.00])
cbar.set_ticklabels(['100% Decrease', 'Same', '100% Increase'])
cbar.ax.tick_params(labelsize=13)

if xDim > 6:
  thisFontSize=9
else:
  thisFontSize=18

# add numbers in each cell
for xstri in range(np.size(TableXstrs)):
    for vari in range(0, np.size(TableVars)):

            # retrieve cell value
            number = np.around(BIASarray[vari,xstri], decimals=2)
            localcolor = np.around(BIASarrayDiffRatios[vari,xstri], decimals=2)

            # place a black number if the cell is a light colour
            if (localcolor >= -0.05) and (localcolor <= 0.05) :
                text = ax.text(xstri-0.28, vari-0.08, str(number), c='black', fontsize=thisFontSize)
            else:
                text = ax.text(xstri-0.28, vari-0.08, str(number), c='white', fontsize=thisFontSize)
                text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

# SAVE THE RMSE PLOT
SaveFolder = VOLNAME+'/ThesisPlots/'

if not os.path.exists(SaveFolder):
    os.makedirs(SaveFolder)

SaveFile = 'TQUVHwindStoplightDiagramBIAS' + str(MinAlt) + 'mTo' + str(MaxAlt) + 'mV2.'+FILEOUTTYPE

plt.savefig( SaveFolder + SaveFile, facecolor='w', bbox_inches='tight', pad_inches=0.5, dpi=300)
cmz_writing_fig(SaveFolder + SaveFile,11)
plt.close()
# In[ ]:





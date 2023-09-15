#!/bin/bash

## This script will take the output from the Python and organize the figures for the publication

OUTTYPE=pdf

cd /Users/cmz5202/NetCDF/METEO600SAG/ThesisPlots/

mkdir -p Figures
rm -fv ./Figures/*

# Figure 1
cp -v UMeanProfiles1DayLeadx001x101_2500mSansTitleInsideLegend.${OUTTYPE} Figures/
cp -v VMeanProfiles1DayLeadx001x101_2500mSansTitleSansLegend.${OUTTYPE} Figures/
cp -v UpWp_MeanProfiles1DayLeadx001x101_2500mTopSansTitleSanLegend.${OUTTYPE} Figures/
cp -v VpWp_MeanProfiles1DayLeadx001x101_2500mTopSansTitleSanLegend.${OUTTYPE} Figures/

# Figure 2
cp -v UMeanProfiles1DayLeadx001x101_2500mSansTitleInsideLegend.${OUTTYPE} Figures/
cp -v VMeanProfiles1DayLeadx001x101_2500mSansTitleSansLegend.${OUTTYPE} Figures/
cp -v HwindMeanProfiles1DayLeadx001x101_2500mSansTitleSansLegend.${OUTTYPE} Figures/
cp -v UBiasProfiles1DayLeadx001x101_2500mSansTitleSansLegend.${OUTTYPE} Figures/
cp -v VBiasProfiles1DayLeadx001x101_2500mSansTitleSansLegend.${OUTTYPE} Figures/
cp -v HwindBiasProfiles1DayLeadx001x101_2500mSansTitleSansLegend.${OUTTYPE} Figures/
cp -v URMSEProfiles1DayLeadx001x101_2500mSansTitleSansLegend.${OUTTYPE} Figures/
cp -v VRMSEProfiles1DayLeadx001x101_2500mSansTitleSansLegend.${OUTTYPE} Figures/
cp -v HwindRMSEProfiles1DayLeadx001x101_2500mSansTitleSansLegend.${OUTTYPE} Figures/

# Figure 3
cp -v UpgradientFluxMapx001aSansXlabel.png Figures/
cp -v UpgradientFluxMapx101bSansXlabel.png Figures/

# Figure 4
cp -v thetaMeanProfiles1DayLeadx001x101_2500mSansTitleInsideLegend.${OUTTYPE} Figures/
cp -v QMeanProfiles1DayLeadx001x101_2500mSansTitleSansLegend.${OUTTYPE} Figures/

# Figure 5
cp -v UMeanProfiles1DayLeadx001x204_2500mSansTitleInsideLegend.${OUTTYPE} Figures/
cp -v VMeanProfiles1DayLeadx001x204_2500mSansTitleSansLegend.${OUTTYPE} Figures/
cp -v UpWp_MeanProfiles1DayLeadx001x204_2500mTopSansTitleSanLegend.${OUTTYPE} Figures/
cp -v VpWp_MeanProfiles1DayLeadx001x204_2500mTopSansTitleSanLegend.${OUTTYPE} Figures/

#Figure 6
cp -v thetaMeanProfiles1DayLeadx001x204_2500mSansTitleInsideLegend.${OUTTYPE} Figures/
cp -v QMeanProfiles1DayLeadx001x204_2500mSansTitleSansLegend.${OUTTYPE} Figures/
cp -v thetaBiasProfiles1DayLeadx001x204_2500mSansTitleSansLegend.${OUTTYPE} Figures/
cp -v QBiasProfiles1DayLeadx001x204_2500mSansTitleSansLegend.${OUTTYPE} Figures/
cp -v thetaRMSEProfiles1DayLeadx001x204_2500mSansTitleSansLegend.${OUTTYPE} Figures/
cp -v QRMSEProfiles1DayLeadx001x204_2500mSansTitleSansLegend.${OUTTYPE} Figures/

# Figure 7
cp -v U_TimeHeightBias1DayLeadx001LT.${OUTTYPE} Figures/
cp -v V_TimeHeightBias1DayLeadx001LT.${OUTTYPE} Figures/
cp -v Hwind_TimeHeightBias1DayLeadx001LT.${OUTTYPE} Figures/
cp -v U_TimeHeightBias1DayLeadx101LT.${OUTTYPE} Figures/
cp -v V_TimeHeightBias1DayLeadx101LT.${OUTTYPE} Figures/
cp -v Hwind_TimeHeightBias1DayLeadx101LT.${OUTTYPE} Figures/
cp -v U_TimeHeightBias1DayLeadx201LT.${OUTTYPE} Figures/
cp -v V_TimeHeightBias1DayLeadx201LT.${OUTTYPE} Figures/
cp -v Hwind_TimeHeightBias1DayLeadx201LT.${OUTTYPE} Figures/
cp -v U_TimeHeightBias1DayLeadx202LT.${OUTTYPE} Figures/
cp -v V_TimeHeightBias1DayLeadx202LT.${OUTTYPE} Figures/
cp -v Hwind_TimeHeightBias1DayLeadx202LT.${OUTTYPE} Figures/
cp -v U_TimeHeightBias1DayLeadx203LT.${OUTTYPE} Figures/
cp -v V_TimeHeightBias1DayLeadx203LT.${OUTTYPE} Figures/
cp -v Hwind_TimeHeightBias1DayLeadx203LT.${OUTTYPE} Figures/
cp -v U_TimeHeightBias1DayLeadx204LT.${OUTTYPE} Figures/
cp -v V_TimeHeightBias1DayLeadx204LT.${OUTTYPE} Figures/
cp -v Hwind_TimeHeightBias1DayLeadx204LT.${OUTTYPE} Figures/

# Figure 8

cp -v Hwind_TimePlotLT200mTO2000mx001x204SansLegend.${OUTTYPE} Figures/
cp -v U_TimePlotLT200mTO2000mx001x204SansLegend.${OUTTYPE} Figures/
cp -v V_TimePlotLT200mTO2000mx001x204SansLegend.${OUTTYPE} Figures/

# Figure 9
cp -v UpgradientFluxMapx101aSansXlabel.png Figures/
cp -v UpgradientFluxMapx201bSansXlabel.png Figures/
cp -v UpgradientFluxMapx202cSansXlabel.png Figures/
cp -v UpgradientFluxMapx203dSansXlabel.png Figures/
cp -v UpgradientFluxMapx204eSansXlabel.png Figures/

# Figure 10/11
cp -v TQUVHwindStoplightDiagramRMSE200mTo2000m.${OUTTYPE} Figures/
cp -v TQUVHwindStoplightDiagramBIAS200mTo2000mV2.${OUTTYPE} Figures/

# Figure 12
cp -v x101UPWPComponentProfilesValtx10SansLegend.${OUTTYPE} Figures/
cp -v x101VPWPComponentProfilesValtx10SansLegend.${OUTTYPE} Figures/
cp -v x204UPWPComponentProfilesValtx10RightLegend.${OUTTYPE} Figures/
cp -v x204VPWPComponentProfilesValtx10RightLegend.${OUTTYPE} Figures/

cd Figures

for FILE in ./*.pdf; do
  pdfcrop "${FILE}" "${FILE}"
done

rm -v /Users/cmz5202/icloud/LaTeX/Graap_GMD/Figures/*
cp -v *png *pdf /Users/cmz5202/icloud/LaTeX/Graap_GMD/Figures/

#TODO
#Remove XLabel from relevant UpgradientFlux plots
#Correctly label Upgradient plots

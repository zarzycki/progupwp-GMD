#!/bin/bash

##############################################################################################################
# This script is an example of batch downloading from DataCommons and organizing into folder structure
# needed by the batch.sh script to generate the intermediate data and figures
# Note that the Stephens soundings from the Zenodo repo also need to be placed in the DATA folder
##############################################################################################################

#### Download data from DataCommons DOI link
wget -r -np -nd --no-use-server-timestamps -A ".zip" "https://www.datacommons.psu.edu/download/meteorology/graap-zarzycki-using-eurec4a-atomic-field-campaign-data-to-improve-trade-wind-regimes-in-the-community-amosphere-model-2023/"

#### Unzip data, touching in case we are on a system that auto purges
for f in *.zip; do
  tempdir="./tmp/"
  echo $tempdir

  mkdir -p -v $tempdir

  unzip "$f" -d "$tempdir"

  find "$tempdir" -type f -exec touch {} \;

  rsync -av --remove-source-files "$tempdir"/ .

  rm -rf "$tempdir"
done

#### Organize untarred data
mkdir -p DATA/LargeDomainCESMoutput

for dir in cesm-x*; do
  if [[ -d "$dir" ]]; then
    subdir=$(find "$dir" -mindepth 2 -maxdepth 2 -type d -print -quit)
    subdir=$(basename "$subdir")

    rsync -av --remove-source-files "$dir/LargeDomainCESMoutput/$subdir/" "DATA/LargeDomainCESMoutput/$subdir/"

    find "$dir" -type d -empty -delete
  fi
done




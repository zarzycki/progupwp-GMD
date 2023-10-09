#!/bin/bash

find /storage/home/cmz5202/scratch/progupwp/DATA/LargeDomainCESMoutput/ -name "*FHIST-ne30-ATOMIC-ERA5-x???.cam.h3.*nc" | while read -r file; do
  echo -e "$file"
  ncl add-PMID.ncl 'fname="'$file'"'
done
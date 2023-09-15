# Graap and Zarzycki, GMD


To reproduce results, first create a conda environment.

```
conda env create -f graap.yml
```

Create a working directory (e.g., `/Users/mydir/progupwp/`)

Then ensure your tree looks something like this with the untarred files accompanying this document:

```
├── DATA
│   ├── LargeDomainCESMoutput
│   │   ├── x001
│   │   │   ├── LeadDay0
│   │   │   ├── LeadDay1
│   │   │   ├── LeadDay2
│   │   │   ├── LeadDay3
│   │   │   └── h4
│   │   │       └── LeadDay1
│   │   ├── x101
│   │   │   ├── LeadDay0
│   │   │   ├── LeadDay1
│   │   │   ├── LeadDay2
│   │   │   ├── LeadDay3
│   │   │   └── h4
│   │   │       └── LeadDay1
│   │   ├── x201
│   │   │   ├── LeadDay0
│   │   │   ├── LeadDay1
│   │   │   ├── LeadDay2
│   │   │   ├── LeadDay3
│   │   │   └── h4
│   │   │       └── LeadDay1
│   │   ├── x202
│   │   │   ├── LeadDay0
│   │   │   ├── LeadDay1
│   │   │   ├── LeadDay2
│   │   │   ├── LeadDay3
│   │   │   └── h4
│   │   │       └── LeadDay1
│   │   ├── x203
│   │   │   ├── LeadDay0
│   │   │   ├── LeadDay1
│   │   │   ├── LeadDay2
│   │   │   ├── LeadDay3
│   │   │   └── h4
│   │   │       └── LeadDay1
│   │   └── x204
│   │       ├── LeadDay0
│   │       ├── LeadDay1
│   │       ├── LeadDay2
│   │       ├── LeadDay3
│   │       └── h4
│   │           └── LeadDay1
│   └── StephanSoundings
│       └── OriginalDownloads
├── SCRIPTS
│   └── python_util
```

Then edit `./batch.sh` in the `SCRIPTS` dir to point to this directory (e.g., `BASEDIR=/Users/mydir/progupwp/`). Specify the number of CPUs on your system (up to 6, set `N=1` for serial) to generate the data and figures.

To verify success, the `BASEDIR` should now include three additional folders:

```
ThesisData
ThesisVariables
ThesisPlots
```

On a 2022 Macbook Pro M1, running with $N=6$ took approximately 4 hours from start to finish.

### Processing datacommons data

```
ZIPDIR=/storage/home/cmz5202/scratch/meteorology/graap-zarzycki-using-eurec4a-atomic-field-campaign-data-to-improve-trade-wind-regimes-in-the-community-amosphere-model-2023/
TARGETDIR=/storage/home/cmz5202/scratch/progupwp/DATA/LargeDomainCESMoutput/
mv -v $ZIPDIR/cesm-x001/LargeDomainCESMoutput/* $TARGETDIR
mv -v $ZIPDIR/cesm-x101/LargeDomainCESMoutput/* $TARGETDIR
mv -v $ZIPDIR/cesm-x201/LargeDomainCESMoutput/* $TARGETDIR
mv -v $ZIPDIR/cesm-x202/LargeDomainCESMoutput/* $TARGETDIR
mv -v $ZIPDIR/cesm-x203/LargeDomainCESMoutput/* $TARGETDIR
mv -v $ZIPDIR/cesm-x204/LargeDomainCESMoutput/* $TARGETDIR
```


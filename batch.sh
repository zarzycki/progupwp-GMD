#!/bin/bash -l
#SBATCH --nodes=1 
#SBATCH --ntasks=12
#SBATCH --mem=100GB
#SBATCH --time=12:00:00
#SBATCH --partition=open

date >> timing.txt

BASEDIR=/storage/home/cmz5202/scratch/progupwp/
N=10
configs=("x001" "x101" "x201" "x202" "x203" "x204" "x301" "x302" "x303" "x304")

conda activate graap

# Loop over and process raw CAM data
for i in "${configs[@]}" ; do
  (
    echo $i
    #sleep $((RANDOM % 10))
    python DataInterpolator.py $BASEDIR $i
    python StateVariablePackager3hour.py $BASEDIR $i
    python StateVariablePackager24hour.py $BASEDIR $i
    python StephanSoundingGroupingModelsOnly.py $BASEDIR $i
  ) &

  # allow to execute up to $N jobs in parallel
  if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
      # now there are $N jobs already running, so wait here for any job
      # to be finished so there is a place to start next one.
      wait -n
  fi
done

# no more jobs to be started but wait for pending jobs
# (all need to be finished)
wait

# Extract turbulence budgets for two runs
python OtherVariablePackager.py $BASEDIR x101
python OtherVariablePackager.py $BASEDIR x204

# Generate figures
python IncludedFigureGenerator.py $BASEDIR

date >> timing.txt
echo "-------" >> timing.txt

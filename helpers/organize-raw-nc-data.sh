set -e

EXP=x302

mkdir -p /glade/u/home/zarzycki/scratch/ATOMIC_RUNS/$EXP
mkdir -p /glade/u/home/zarzycki/scratch/ATOMIC_RUNS/$EXP/h4/LeadDay1
mkdir -p /glade/u/home/zarzycki/scratch/ATOMIC_RUNS/$EXP/LeadDay0
mkdir -p /glade/u/home/zarzycki/scratch/ATOMIC_RUNS/$EXP/LeadDay1
mkdir -p /glade/u/home/zarzycki/scratch/ATOMIC_RUNS/$EXP/LeadDay2
mkdir -p /glade/u/home/zarzycki/scratch/ATOMIC_RUNS/$EXP/LeadDay3

cd /glade/u/home/zarzycki/scratch/ATOMIC_RUNS/FHIST-ne30-ATOMIC-ERA5-$EXP

for d in 2020*00/ ; do
  echo $d
  # Remove the trailing "/"
  input_str=${d%/}

  # Extract components
  YYYY=${input_str:0:4}
  MM=${input_str:4:2}
  DD=${input_str:6:2}
  HH=${input_str:8:2}

  # Add one days and get the new date and time components
  day0date=$(date -d "$YYYY-$MM-$DD $HH:00:00 0 days" '+%Y-%m-%d-%H')
  IFS="-" read -ra parts <<< "$day0date"
  seconds_in_day=$(printf "%05d" $((${parts[3]} * 3600)))
  final_day0="${parts[0]}-${parts[1]}-${parts[2]}-$seconds_in_day"
  echo $final_day0
  cp -v $d/*.cam.h3.$final_day0.nc /glade/u/home/zarzycki/scratch/ATOMIC_RUNS/$EXP/LeadDay0/

  # Add one days and get the new date and time components
  day1date=$(date -d "$YYYY-$MM-$DD $HH:00:00 1 days" '+%Y-%m-%d-%H')
  IFS="-" read -ra parts <<< "$day1date"
  seconds_in_day=$(printf "%05d" $((${parts[3]} * 3600)))
  final_day1="${parts[0]}-${parts[1]}-${parts[2]}-$seconds_in_day"
  echo $final_day1
  cp -v $d/*.cam.h3.$final_day1.nc /glade/u/home/zarzycki/scratch/ATOMIC_RUNS/$EXP/LeadDay1/
  cp -v $d/*.cam.h4.$final_day1.nc /glade/u/home/zarzycki/scratch/ATOMIC_RUNS/$EXP/h4/LeadDay1

  # Add one days and get the new date and time components
  day2date=$(date -d "$YYYY-$MM-$DD $HH:00:00 2 days" '+%Y-%m-%d-%H')
  IFS="-" read -ra parts <<< "$day2date"
  seconds_in_day=$(printf "%05d" $((${parts[3]} * 3600)))
  final_day2="${parts[0]}-${parts[1]}-${parts[2]}-$seconds_in_day"
  echo $final_day2
  cp -v $d/*.cam.h3.$final_day2.nc /glade/u/home/zarzycki/scratch/ATOMIC_RUNS/$EXP/LeadDay2/

  # Add one days and get the new date and time components
  day3date=$(date -d "$YYYY-$MM-$DD $HH:00:00 3 days" '+%Y-%m-%d-%H')
  IFS="-" read -ra parts <<< "$day3date"
  seconds_in_day=$(printf "%05d" $((${parts[3]} * 3600)))
  final_day3="${parts[0]}-${parts[1]}-${parts[2]}-$seconds_in_day"
  echo $final_day3
  cp -v $d/*.cam.h3.$final_day3.nc /glade/u/home/zarzycki/scratch/ATOMIC_RUNS/$EXP/LeadDay3/

done



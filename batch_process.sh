for file in  samples/*.jpg
do
  python starrem2k13.py  "$file" "samples_starless/${file##*/}"
done
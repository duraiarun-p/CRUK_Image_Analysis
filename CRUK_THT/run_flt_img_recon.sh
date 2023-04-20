#/usr/bin/bash
echo "The shell script has been initiated"
echo "Sourcing the conda"
source ~/anaconda3/etc/profile.d/conda.sh
echo "Initiating conda env im_py37"
conda activate im_py37
echo "Env im_py37 activated"
echo "Current working directory"
pwd
touch normal_log.txt
cd /mnt/local_share/TMA/FS-FLIM/raw/Normal_1B
#ls -l
MYPATH_NORMAL=$(pwd)
#$MYPATH_NORMAL="/mnt/local_share/TMA/FS-FLIM/raw/Normal_1B"
echo $MYPATH_NORMAL
cd $MYPATH_NORMAL
for NORMAL in R*
do
 cd $NORMAL
 echo $NORMAL >> /home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/CRUK_THT/normal_log.txt
 MYPATH_NORMAL_CORE=$(pwd)
 mkdir -p FLT_IMG_DIR
 FLT_IMG_DIR_NAME=FLT_IMG_DIR
 MYPATH_NORMAL_CORE_OUTPUT="$MYPATH_NORMAL_CORE/$FLT_IMG_DIR_NAME"
 echo $NORMAL
 echo "Executing Python Script"
 python /home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/CRUK_THT/flt_img_recon_sys.py $MYPATH_NORMAL_CORE $MYPATH_NORMAL_CORE_OUTPUT
 cd ..
 
done
    
#TEST_NORMAL_CORE="/mnt/local_share/TMA/FS-FLIM/raw/Normal_1B/Row-1_Col-1_20230303/"
#TEST_NORMAL_OUTPUT="/mnt/local_share/TMA/FS-FLIM/raw/Normal_1B/Row-1_Col-1_20230303/FLT_IMG_DIR"
#echo "Executing Python Script"
#python /home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/CRUK_THT/flt_img_recon_sys.py $TEST_NORMAL_CORE $TEST_NORMAL_OUTPUT

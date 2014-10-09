#EBLEARN simple script for Bootstrapping the NN 
 
if test $# -ne 1 || test $1 -gt 3 || test $1 -lt 0; then 
echo " usage : $0 [NUMBER OF CYCLES (0-3)] " 
exit -1
fi


#add bootstrapping lines to the config only if not already present 

if ! cat face.conf | grep -q "BOOTSTRAPPING VARIABLES"; then   

echo "# BOOTSTRAPPING VARIABLES ##########################################
weights=net00030.mat
classes=${val_classes}
input_dir=data/bg/bgimgs
# boostrapping
bootstrapping = 1          # enable bootstrapping extraction or not
bootstrapping_save = 1     # save bootstrapped samples directly as a dataset
display_bootstrapping = 0  # display bootstrapping extraction
display = 0
display_sleep = 0
gt_neg_max = 3		   # maximum number of negatives to be saved per classifier and image

# negative bootstrapping
bootstrapping_max = 5000   # limit to this number of extracted samples (optional)
gt_extract_pos = 0         # extract positive samples or not
gt_extract_neg = 1         # extract negative samples or not
gt_neg_threshold = .01     # minimum confidence of extracted negative samples
gt_neg_gt_only = 0         # only extract negatives when positives are present in image
input_random = 1           # taking random images is better for bootstrapping
gt_name = face_neg    # name of saved bootstrapping dataset
bbox_scalings = 1x1        # scaling factors of detected boxes
output_dir=bootstrap_output
# END OF BOOTSTRAPPING VARIABLES ########################################
" >> face.conf

fi 


cp face.conf face.conf.new.0   #just for easiness for the first iteration 

for ((i = 0; i < $1; i += 1)); 
do 

echo "bootstrapping loop $i" >> log.bootstrapping


detect face.conf

#I know this sucks but I didn't have time to fix it 

case $i in
1) 
    first=""
    second="+fp1"
  ;;
2) 
 first="+fp1" 
 second="+fp1+fp2" 
  ;;
3) 

first="+fp1+fp2"
second="+fp1+fp2+fp3"

;;

*) echo "error" 
;; 
esac 

rm -rf  prepared_data/bootstrap*.mat
mv bootstrap_output/detections*/bootstrapping_face_neg_*.mat prepared_data/
rm -rf bootstrap_output
dsmerge prepared_data face+bg_train$second face+bg_train$first bootstrapping_face_neg


sed 's|train_dsname=face+bg_train$first|train_dsname=face+bg_train$second|' <face.conf.new.$(( $i-1 )) >face.conf.new.$i

#train the net with new face.conf file 
train face.conf.new.$i

done 

#As it finish restore the config original file, later you can test your improvements launching  "detect face.conf" 
cp face.conf.backup face.conf



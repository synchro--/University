RANK=$1
METHOD=$2
MODEL=$3
LAYER=$4

python dump_layer.py $MODEL $LAYER
cd dumps; matlab -nodesktop -nojvm -nodisplay -r "compute_cpd('weights.mat', $RANK, $METHOD);"
mv out_file.mat zhang.mat; cd .. 

function usage() {
	echo Usage: $0 '<number of epochs>'
	exit -1
}


if [[ $# -lt 1 ]]; then 
	usage 
fi 

EPOCHS=$1
th resnet-tuning.lua -nEpochs $EPOCHS
python plot_log.py


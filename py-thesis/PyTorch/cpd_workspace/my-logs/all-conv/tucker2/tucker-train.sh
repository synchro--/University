## Training with Tucker Decomposition

# python main.py --decompose --layer 'conv1'  
# python main.py --fine-tune
# mv cifar10.csv cp-tucker-1.csv 
# cp s_trained.pth ./saved_models/s_cp-tucker-1.pth

python main.py --decompose --layer 'conv2'  
python main.py --fine-tune
mv cifar10.csv cp-tucker-12.csv 
cp s_trained.pth ./saved_models/s_cp-tucker-12.pth

python main.py --decompose --layer 'conv3'  
python main.py --fine-tune
mv cifar10.csv cp-tucker-123.csv 
cp s_trained.pth ./saved_models/s_cp-tucker-123.pth

python main.py --decompose --layer 'conv4'  
python main.py --fine-tune
mv cifar10.csv cp-tucker-1234.csv 
cp s_trained.pth ./saved_models/s_cp-tucker-1234.pth

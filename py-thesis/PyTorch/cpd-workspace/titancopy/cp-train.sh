python main.py --fine-tune --cp 
mv cifar10.csv cp-cl-BN-1.csv 

python main.py --decompose --cp --layer 'conv2'  
python main.py --fine-tune --cp 
mv cifar10.csv cp-cl-BN-12.csv 
cp s_trained.pth ./saved_models/s_cp-cl-BN-12.pth

python main.py --decompose --cp --layer 'conv3'  
python main.py --fine-tune --cp 
mv cifar10.csv cp-cl-BN-123.csv 
cp s_trained.pth ./saved_models/s_cp-cl-BN-123.pth

python main.py --decompose --cp --layer 'conv4'  
python main.py --fine-tune --cp 
mv cifar10.csv cp-cl-BN-all.csv 
cp s_trained.pth ./saved_models/s_cp-cl-BN-all.pth
cp decomposed_model.pth ./saved_models/m_cp-cl-BN-all.pth





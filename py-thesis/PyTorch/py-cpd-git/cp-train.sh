python main.py --fine-tune --cp 
mv cifar10.csv cp-TF-cl-1.csv 

python main.py --decompose --cp --layer 'conv2'  
python main.py --fine-tune --cp 
mv cifar10.csv cp-TF-cl-12.csv 
cp s_trained.pth ./saved_models/s_cp-TF-cl-12.pth

python main.py --decompose --cp --layer 'conv3'  
python main.py --fine-tune --cp 
mv cifar10.csv cp-TF-cl-123.csv 
cp s_trained.pth ./saved_models/s_cp-TF-cl-123.pth

python main.py --decompose --cp --layer 'conv4'  
python main.py --fine-tune --cp 
mv cifar10.csv cp-TF-cl-all.csv 
cp s_trained.pth ./saved_models/s_cp-TF-cl-all.pth
cp decomposed_model.pth ./saved_models/m_cp-TF-cl-all.pth





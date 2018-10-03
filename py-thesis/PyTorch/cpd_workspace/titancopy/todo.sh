python main.py --decompose --cp --fine-tune --model s_allconv.pth
mv *reverse*.csv ./exp1/
mv *reverse*.pth ./exp1/ 


python main.py --decompose --fine-tune --model s_allconv.pth
mv *reverse*.csv ./exp2/
mv *reverse*.pth ./exp2/ 



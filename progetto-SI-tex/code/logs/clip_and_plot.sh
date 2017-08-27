for i in *.log; do 
	ROWS=$(cat $i | wc -l) 
	if [ $ROWS -gt 201 ]; then 
		head -201 $i > tmp 
		mv tmp $i
		
	fi
done

python plot_comparison.py

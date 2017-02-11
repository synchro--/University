#this script enable you to label your dataset accounting to your folders and then use Torch7 tensor syntax to load them quicly 
echo " Run this on the parent directory of your photo folders
for example: run in photo/ and the folders are named : photo/person photo/tree etc"

if test $# -ne 1 ; then 
 echo " Usage: labeling.sh directory"
 exit 1 
fi
 
cd $1
 
for i in *; do
  count=1
  if test -d "$i"; then
   prefix=$(basename "$i")
   cd $i 
   for j in *; do
    if test -f "$j"; then 
    mv "$j" $prefix.$count.png
    count=$[ $count +1 ]
    fi 
    echo "$prefix" >> label.txt
    echo $count >> count.txt 
   done
   cd ..
  fi
 done
 
 echo "Naming and labeled completed" 
 echo "Now you have for every folder all the name regular, a label.txt file that contains the name of the label and a count.txt with the number of files present in the directory"
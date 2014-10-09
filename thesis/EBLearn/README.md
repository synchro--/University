This is only a trivial script to running the ripetitive steps on the 3rd tutorial of EBLearn. (http://eblearn.sourceforge.net/beginner_tutorial3_face_detector.html) 

Given the face.conf file (the one in this directory is fine) and the downloaded datasets (see first step of the link above) the script creates the datasets using the dscompile tool, trains the net and does the bootstrapping. These steps are repeated the number of times specified by the first parameter. The first parameter can be max 3 

For example : ./bootstrapping.sh 3 #makes it repeat the steps 3 times

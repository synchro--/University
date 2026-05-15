Testing code for paper "Learning from scratch a confidence measure", BMVC 2016.
If you use this code, please cite this paper:

M. Poggi, S. Mattoccia, “Learning from scratch a conﬁdence measure”, accepted at 27th British Machine Vision Conference (BMVC 2016), September 19-22, 2016, York, UK 

Bibtex:

@INPROCEEDINGS{CCNN_BMVC_2016,
   author = “Poggi, {M} and Mattoccia, {S}",
   title = "Learning from scratch a conﬁdence measure",
   booktitle = "27th British Machine Vision Conference (BMVC 2016)",
   month = Sep,
   year = 2016,
   day = “19-22”,
   address = “York, UK”,
}



Hardware Requirements:
	A PC with a standard CPU (soon a GPU version of this code)
	At least 4GB RAM (the test requires about 2GB memory to run)

Software Requirements: 
	Linux
	Torch 7 (torch.ch)
	*OpenCV 2.4.x (opencv.org)

Usage:
	Open a shell and compile merge utility by running `make` to enable 16 bit confidence encoding.
	To test the CNN, run `th BMVC16_test.lua`.
	Default parameters will depict the confidence map for the provided disparity sample.
	Run `th BMVC16_test.lua -disparity X.png -output Y.png -enable16Bit 'false'` if you want to run the demo on a different disparity map named X.png, saving the confidence as Y.png. Setting -enable16Bit to 'false' will save confidence maps as 8 bit images (lower precision), but will not required OpenCV to be installed.

*OpenCV is required for 16-bit confidence maps, otherwise only Torch is required.

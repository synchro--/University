#include "cv.h"
#include "highgui.h"

int main(int argc, char **argv)
{
	if (argc != 4)
	{
		printf("Usage: %s <map H> <map L> <output>\n", argv[0]);
	}
	IplImage *H = cvLoadImage(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	IplImage *L = cvLoadImage(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
	IplImage *C = cvCreateImage(cvGetSize(H), IPL_DEPTH_16U, 1);

	for (int i = 0; i < H->height; i++)
		for (int j = 0; j < H->width; j++)
		{
			CV_IMAGE_ELEM(C, ushort, i, j) = CV_IMAGE_ELEM(H, uchar, i, j) * 256 + CV_IMAGE_ELEM(L, uchar, i, j);
		}
	cvSaveImage(argv[3], C);
	cvReleaseImage(&H);
	cvReleaseImage(&L);
	cvReleaseImage(&C);
	return 0;
}	

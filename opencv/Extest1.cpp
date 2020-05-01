#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
int flag = 0;

using namespace cv;
using std::cout;
using std::endl;
using std::vector;
using std::string;

void calc_2D_entropy(cv::Mat &input, cv::Mat &output) {
	int height = input.rows;
	int width = input.cols;
	cv::Mat out = cv::Mat::zeros(height, width, CV_32FC1);
	//template size
	int w = 3;

	for (int i = w; i < height - w; i++)
	{
		float *data = out.ptr<float>(i);
		for (int j = w; j < width - w; j++)
		{
			//cv::Mat Hist = cv::Mat::zeros(1, 256, CV_32F);
			float Hist[256] = { 0 };
			for (int p = i - w; p < i + w + 1; p++)
			{
				uchar *t = input.ptr<uchar>(p);
				for (int q = j - w; q < j + w + 1; q++)
				{
					int tmp = t[q];
					//cout << "tmp:" << tmp << endl;
					Hist[tmp] = Hist[tmp] + 1;
				}

			}

			float sumHist = 0;
			for (int ii = 0; ii < 256; ii++)
			{

				sumHist += Hist[ii];
			}

			//get the probality
			for (int ii = 0; ii < 256; ii++)
			{
				Hist[ii] = Hist[ii] / sumHist;
				//if (Hist[ii] != 0)
				//  cout << ii << ":" << Hist[ii] << endl;
			}

			//calculate the entropy
			for (int k = 0; k < 256; k++)
			{
				float v = Hist[k];
				float z = data[j];
				//cout << "z:" << z << endl;
				if (v != 0)
				{
					double H = v * (log(v) / (float)log(2.0));
					//H = H * 80.5 - 1;
					data[j] = data[j] - H;
					//data[j] = data[j] + v * log(1 / v);
					//cout << j << ":" << data[j] << endl;
				}
			}
		}

	}

	normalize(out, output);
	output = output * 255;

}
static int32_t sub_to_ind(int32_t *coords, int32_t *cumprod, int32_t num_dims)
{
	int index = 0;
	int k;

	assert(coords != NULL);
	assert(cumprod != NULL);
	assert(num_dims > 0);

	for (k = 0; k < num_dims; k++)
	{
		index += coords[k] * cumprod[k];
	}

	return index;
}
static void ind_to_sub(int p, int num_dims, const int size[], int *cumprod, int *coords)
{
	int j;

	assert(num_dims > 0); 
	assert(coords != NULL);
	assert(cumprod != NULL);

	for (j = num_dims - 1; j >= 0; j--)
	{
		coords[j] = p / cumprod[j];
		p = p % cumprod[j];
	}
}
void getLocalEntropyImage(cv::Mat &gray, cv::Rect &roi, cv::Mat &entropy)
{
	clock_t func_begin, func_end;
	func_begin = clock();
	//1.define nerghbood model,here it's 9*9
	int neighbood_dim = 2;
	int neighbood_size[] = { 9, 9 };

	//2.Pad gray_src
	Mat gray_src_mat(gray);
	Mat pad_mat;
	int left = (neighbood_size[0] - 1) / 2;
	int right = left;
	int top = (neighbood_size[1] - 1) / 2;
	int bottom = top;
	copyMakeBorder(gray_src_mat, pad_mat, top, bottom, left, right, BORDER_REPLICATE, 0);
	Mat *pad_src = &pad_mat;
	roi = cv::Rect(roi.x + top, roi.y + left, roi.width, roi.height);

	//3.initial neighbood object,reference to Matlab build-in neighbood object system
	//        int element_num = roi_rect.area();
	//here,implement a histogram by ourself ,each bin calcalate gray value frequence
	int hist_count[256] = { 0 };
	int neighbood_num = 1;
	for (int i = 0; i < neighbood_dim; i++)
		neighbood_num *= neighbood_size[i];

	//neighbood_corrds_array is a neighbors_num-by-neighbood_dim array containing relative offsets
	int *neighbood_corrds_array = (int *)malloc(sizeof(int)*neighbood_num * neighbood_dim);
	//Contains the cumulative product of the image_size array;used in the sub_to_ind and ind_to_sub calculations.
	int *cumprod = (int *)malloc(neighbood_dim * sizeof(*cumprod));
	cumprod[0] = 1;
	for (int i = 1; i < neighbood_dim; i++)
		cumprod[i] = cumprod[i - 1] * neighbood_size[i - 1];
	int *image_cumprod = (int*)malloc(2 * sizeof(*image_cumprod));
	image_cumprod[0] = 1;
	image_cumprod[1] = pad_src->cols;
	//initialize neighbood_corrds_array
	int p;
	int q;
	int *coords;
	for (p = 0; p < neighbood_num; p++) {
		coords = neighbood_corrds_array + p * neighbood_dim;
		ind_to_sub(p, neighbood_dim, neighbood_size, cumprod, coords);
		for (q = 0; q < neighbood_dim; q++)
			coords[q] -= (neighbood_size[q] - 1) / 2;
	}
	//initlalize neighbood_offset in use of neighbood_corrds_array
	int *neighbood_offset = (int *)malloc(sizeof(int) * neighbood_num);
	int *elem;
	for (int i = 0; i < neighbood_num; i++) {
		elem = neighbood_corrds_array + i * neighbood_dim;
		neighbood_offset[i] = sub_to_ind(elem, image_cumprod, 2);
	}

	//4.calculate entroy for pixel
	uchar *array = (uchar *)pad_src->data;
	//here,use entroy_table to avoid frequency log function which cost losts of time
	float entroy_table[82];
	const float log2 = log(2.0f);
	entroy_table[0] = 0.0;
	float frequency = 0;
	for (int i = 1; i < 82; i++) {
		frequency = (float)i / 81;
		entroy_table[i] = frequency * (log(frequency) / log2);
	}
	int neighbood_index;
	//        int max_index=pad_src->cols*pad_src->rows;
	float e;
	int current_index = 0;
	int current_index_in_origin = 0;
	for (int y = roi.y; y < roi.height; y++) {
		current_index = y * pad_src->cols;
		current_index_in_origin = (y - 4) * gray.cols;
		for (int x = roi.x; x < roi.width; x++, current_index++, current_index_in_origin++) {
			for (int j = 0; j<neighbood_num; j++) {
				neighbood_index = current_index + neighbood_offset[j];
				hist_count[array[neighbood_index]]++;
			}
			//get entropy
			e = 0;
			for (int k = 0; k < 256; k++) {
				if (hist_count[k] != 0) {
					//                                        int frequency=hist_count[k];
					e -= entroy_table[hist_count[k]];
					hist_count[k] = 0;
				}
			}
			((float *)entropy.data)[current_index_in_origin] = e;
		}
	}
	free(neighbood_offset);
	free(image_cumprod);
	free(cumprod);
	free(neighbood_corrds_array);

	func_end = clock();
	double func_time = (double)(func_end - func_begin) / CLOCKS_PER_SEC;
	std::cout << "func time" << func_time << std::endl;
}
/*int flag(string str)
{
	Mat result3 =imread(str);

	if ((result3.rows)*0.7 - contours[0][0].y *1.0 > 0) return 2;//zhixing
	else if (contours[0][0].x == 0) return 1;//zuo
	else return 3;
}*/

int splitBaseOnColor(string str) {
	clock_t func_begin, func_end;
	func_begin = clock();
	Mat src = imread(str);
	cvtColor(src, src, CV_BGR2HSV);
	Mat dst(src.rows, src.cols, CV_8UC1, cv::Scalar::all(0));
	for (int i = 0; i<src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			if (src.at<cv::Vec3b>(i, j)[0] >= 15 &&
				src.at<cv::Vec3b>(i, j)[0] <= 30 &&
				src.at<cv::Vec3b>(i, j)[1] >= 60 &&
				src.at<cv::Vec3b>(i, j)[1] <= 255 &&
				src.at<cv::Vec3b>(i, j)[2] >= 60 &&
				src.at<cv::Vec3b>(i, j)[2] <= 255
				)
				dst.at<uchar>(i, j) = 255;
			else
				dst.at<uchar>(i, j) = 0;
		}
	medianBlur(dst, dst, 11);
	dilate(dst, dst, getStructuringElement(MORPH_RECT, Size(15, 15)));
	vector<vector<cv::Point> > contours;

	findContours(dst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	double maxArea = 0;
	int subscript = 0;
	for (size_t i = 0; i < contours.size(); i++)
	{
		double area = cv::contourArea(contours[i]);
		if (area > maxArea)
		{
			maxArea = area;
			subscript = i;
		}
	}

	Rect maxRect = boundingRect(contours[subscript]);

	Mat result1(src.rows, src.cols, CV_8UC1, cv::Scalar::all(255));
	for (size_t i = 0; i < contours.size(); i++)
	{
		if (i != subscript)
		{
			Rect rect = boundingRect(contours[i]);
			rectangle(result1, rect, cv::Scalar(0), CV_FILLED);
		}
	}
	Mat result2;
	rectangle(result2, maxRect, cv::Scalar(255), CV_FILLED);
	dst.copyTo(result2, result2);
	//imshow("kshf", result2);
	Mat result3;
	result2.copyTo(result3, result1);
	//imshow("largest region", result3);    
	func_end = clock();
//	double func_time = (double)(func_end - func_begin) / CLOCKS_PER_SEC;
//	std::cout << "func time" << func_time << std::endl;
	//std::cout << flag("result3") << endl;
	//std::cout << contours[0][0].y *1.0 << " " << (result3.rows)*0.3 << endl;
	if (contours[0][0].y *1.0 - (result3.rows)*0.3<0) return 2; 
	else if (contours[0][0].x == 0) return 1;
	else return 3;
	
	/*
	

	std::cout <<"kuan " <<result3.cols << "gao" << result3.rows<<endl;
	//imwrite("res.jpg", result3);

	cv::findContours(
		result3,
		contours,
		cv::noArray(),
		cv::RETR_LIST,
		cv::CHAIN_APPROX_SIMPLE
	);
	
	for(int i=0;i<200;i++)
	std::cout << contours[0][i]<<endl; 


	result3 = cv::Scalar::all(0);
	cv::drawContours(result3, contours, -1, cv::Scalar::all(255));

	//cv::imshow("gray image", image_gray);
	cv::imshow("Contours", result3);
	*/
	//waitKey(0);
}
void splitBaseOnTexture(string str) {
	clock_t func_begin, func_end;
	func_begin = clock();
	//string str = "test7.jpg";
	Mat src = imread(str);
	imshow("largest region", src);
	blur(src, src, Size(3, 3));
	GaussianBlur(src, src, Size(5, 5), 0, 0);
	Mat dst(src.size(), CV_64FC1);
	cvtColor(src, src, COLOR_BGR2GRAY);

									   //getLocalEntropyImage(src,Rect(0,0,src.cols+4,src.rows+4), dst);
	calc_2D_entropy(src, dst);
	////cvtColor(src, src, CV_BGR2YUV);
	//vector<Mat> channels;
	//split(src, channels);

	//dct(Mat_<double>(channels.at(1)), dst);

	normalize(dst, dst, 0, 255, cv::NORM_MINMAX);
	threshold(dst, dst, 210, 1, THRESH_BINARY);

	erode(dst, dst, getStructuringElement(MORPH_RECT, Size(4, 4)));
	dilate(dst, dst, getStructuringElement(MORPH_RECT, Size(13, 13)));

	Mat entropy;
	dst.convertTo(entropy, CV_8U);
	vector<vector<cv::Point> > contours;
	findContours(entropy, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	double maxArea = 0;
	int subscript = 0;
	for (size_t i = 0; i < contours.size(); i++)
	{
		double area = cv::contourArea(contours[i]);
		if (area > maxArea)
		{
			maxArea = area;
			subscript = i;
		}
	}

	Rect maxRect = boundingRect(contours[subscript]);

	Mat result1(src.rows, src.cols, CV_8UC1, cv::Scalar::all(255));
	for (size_t i = 0; i < contours.size(); i++)
	{
		if (i != subscript)
		{
			Rect rect = boundingRect(contours[i]);
			rectangle(result1, rect, cv::Scalar(0), CV_FILLED);
		}
	}
	Mat result2;
	rectangle(result2, maxRect, cv::Scalar(255), CV_FILLED);
	dst.copyTo(result2, result2);
	Mat result3;
	result2.copyTo(result3, result1);
	imshow("Large range", result3);
	dilate(result3, result3, getStructuringElement(MORPH_RECT, Size(15, 15)));
	normalize(result3, result3, 0, 255, cv::NORM_MINMAX);
	
	//imwrite("res-" + str, result3);
	int a = 0;
	func_end = clock();
	double func_time = (double)(func_end - func_begin) / CLOCKS_PER_SEC;
	std::cout << "func time" << func_time << std::endl;
	if (contours[0][0].y *1.0-(result3.rows)*0.3>0) flag=2;
	else if (contours[0][0].x == 0) flag=1;
	else flag=3;
	waitKey(0);
}



int main(int argc ,char *argv[])
{
	int ans = 0;
        flag =  splitBaseOnColor(argv[1]);
	printf("%d\n",flag);
	return 0;
}

int split()
{
	splitBaseOnColor("image.jpg");
	return flag;

}









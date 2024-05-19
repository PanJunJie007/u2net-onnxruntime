#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;


class U2Net
{
public:
	U2Net();
	Mat detect(Mat& cv_image);
private:
	vector<float> m_vInputImageData;
	int m_inpWidth;
	int m_inpHeight;
	int m_outWidth;
	int m_outHeight;
	const float m_mean[3] = { 0.485, 0.456, 0.406 };
	const float m_stds[3] = { 0.229, 0.224, 0.225 };

	Env m_env = Env(ORT_LOGGING_LEVEL_ERROR, "u2net");
	Ort::Session *m_ortSession = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> m_vInputNames;
	vector<char*> m_vOutputNames;
	vector<vector<int64_t>> m_vInputNodeDims; // >=1 outputs
	vector<vector<int64_t>> m_vOutputNodeDims; // >=1 outputs
};

U2Net::U2Net()
{
	string model_path = "models/u2net_portrait.onnx";
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	m_ortSession = new Session(m_env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = m_ortSession->GetInputCount();
	size_t numOutputNodes = m_ortSession->GetOutputCount();
	AllocatorWithDefaultOptions allocator;

	for (int i = 0; i < numInputNodes; i++)
	{
		//m_vInputNames.push_back(m_ortSession->GetInputName(i, allocator));
		AllocatedStringPtr input_name_Ptr = m_ortSession->GetInputNameAllocated(i, allocator);

		int length = strlen(input_name_Ptr.get());
		char* buf = new char[length];
		strcpy(buf, input_name_Ptr.get());

		m_vInputNames.push_back(buf);

		Ort::TypeInfo input_type_info = m_ortSession->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		m_vInputNodeDims.push_back(input_dims);
	}

	for (int i = 0; i < numOutputNodes; i++)
	{
		//m_vOutputNames.push_back(m_ortSession->GetOutputName(i, allocator));
		AllocatedStringPtr output_name_Ptr = m_ortSession->GetOutputNameAllocated(i, allocator);

		int length = strlen(output_name_Ptr.get());
		char* buf = new char[length];
		strcpy(buf, output_name_Ptr.get());

		m_vOutputNames.push_back(buf);

		Ort::TypeInfo output_type_info = m_ortSession->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		m_vOutputNodeDims.push_back(output_dims);
	}

	this->m_inpHeight = m_vInputNodeDims[0][2];
	this->m_inpWidth = m_vInputNodeDims[0][3];
	this->m_outHeight = m_vOutputNodeDims[0][2];
	this->m_outWidth = m_vOutputNodeDims[0][3];
}

Mat U2Net::detect(Mat& srcimg)
{
	Mat dstimg;
	resize(srcimg, dstimg, Size(this->m_inpWidth, this->m_inpHeight));
	this->m_vInputImageData.resize(this->m_inpWidth * this->m_inpHeight * dstimg.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < this->m_inpHeight; i++)
		{
			for (int j = 0; j < this->m_inpWidth; j++)
			{
				float pix = dstimg.ptr<uchar>(i)[j * 3 + 2 - c];
				this->m_vInputImageData[c * this->m_inpHeight * this->m_inpWidth + i * this->m_inpWidth + j] = (pix /255.0 - m_mean[c]) / m_stds[c];
			}
		}
	}
	array<int64_t, 4> input_shape_{ 1, 3, this->m_inpHeight, this->m_inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, m_vInputImageData.data(), m_vInputImageData.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = m_ortSession->Run(RunOptions{ nullptr }, &m_vInputNames[0], &input_tensor_, 1, m_vOutputNames.data(), m_vOutputNames.size());   // 开始推理
	float* pred = ort_outputs[0].GetTensorMutableData<float>();
	Mat result(m_outHeight, m_outWidth, CV_32FC1, pred);
	result = 1 - result;
	double min_value, max_value;
	minMaxLoc(result, &min_value, &max_value, 0, 0);
	result = (result - min_value) / (max_value - min_value);
	result *= 255;
	result.convertTo(result, CV_8UC1);
	return result;
}

int main()
{
	U2Net mynet;
	string imgpath = "sample.jpg";
	Mat srcimg = imread(imgpath);
	Mat result = mynet.detect(srcimg);
	resize(result, result, Size(srcimg.cols, srcimg.rows));

	namedWindow("srcimg", WINDOW_NORMAL);
	imshow("srcimg", srcimg);
	static const string kWinName = "Deep learning object detection in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, result);
	waitKey(0);
	destroyAllWindows();
}
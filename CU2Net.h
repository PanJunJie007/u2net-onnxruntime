#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>

using namespace cv;
using namespace std;
using namespace Ort;

class CU2Net
{
private:
	CU2Net();
	~CU2Net();

public:
	static CU2Net* GetInstance();

public:
	int loadModel(const string& modelPath);
	Mat detect(const Mat& srcimg);

private:
	static CU2Net* m_pInstance;

private:
	vector<float> m_vInputImageData;
	int m_inpWidth;
	int m_inpHeight;
	int m_outWidth;
	int m_outHeight;
	const float m_mean[3] = { 0.485, 0.456, 0.406 };
	const float m_stds[3] = { 0.229, 0.224, 0.225 };

	Env m_env = Env(ORT_LOGGING_LEVEL_ERROR, "u2net");
	Ort::Session* m_ortSession;
	SessionOptions m_sessionOptions;
	vector<char*> m_vInputNames;
	vector<char*> m_vOutputNames;
	vector<vector<int64_t>> m_vInputNodeDims;
	vector<vector<int64_t>> m_vOutputNodeDims;
	string m_sModelPath;
};


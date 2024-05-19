#include "CU2Net.h"

CU2Net* CU2Net::m_pInstance = nullptr;

CU2Net::CU2Net()
{
	m_ortSession = nullptr;
}

CU2Net::~CU2Net()
{
	if (m_ortSession != nullptr)
	{
		m_ortSession = nullptr;
		delete m_ortSession;
	}
}

CU2Net* CU2Net::GetInstance()
{
	if (m_pInstance == nullptr)
	{
		m_pInstance = new CU2Net;
	}

	return m_pInstance;
}

int CU2Net::loadModel(const string& modelPath)
{
	m_sModelPath = modelPath;
	std::wstring widestr = std::wstring(m_sModelPath.begin(), m_sModelPath.end());
	m_sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

	if (m_ortSession != nullptr)
	{
		delete m_ortSession;
		m_ortSession = nullptr;
	}

	for (int i = 0; i < m_vInputNames.size(); i++)
	{
		delete m_vInputNames[i];
			 
	}
	m_vInputNames.clear();

	for (int i = 0; i < m_vOutputNames.size(); i++)
	{
		delete m_vOutputNames[i];

	}
	m_vOutputNames.clear();

	m_ortSession = new Session(m_env, widestr.c_str(), m_sessionOptions);
	size_t numInputNodes = m_ortSession->GetInputCount();
	size_t numOutputNodes = m_ortSession->GetOutputCount();

	AllocatorWithDefaultOptions allocator;

	for (int i = 0; i < numInputNodes; i++)
	{
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

	m_inpHeight = m_vInputNodeDims[0][2];
	m_inpWidth = m_vInputNodeDims[0][3];
	m_outHeight = m_vOutputNodeDims[0][2];
	m_outWidth = m_vOutputNodeDims[0][3];

	return 0;
}

Mat CU2Net::detect(const Mat& srcimg)
{
	Mat dstimg;
	resize(srcimg, dstimg, Size(m_inpWidth, m_inpHeight));
	m_vInputImageData.resize(m_inpWidth * m_inpHeight * dstimg.channels());

	for (int c = 0; c < dstimg.channels(); c++)
	{
		for (int i = 0; i < m_inpHeight; i++)
		{
			for (int j = 0; j < m_inpWidth; j++)
			{
				float pix = dstimg.ptr<uchar>(i)[j * 3 + 2 - c];
				m_vInputImageData[c * m_inpHeight * m_inpWidth + i * m_inpWidth + j] = (pix / 255.0 - m_mean[c]) / m_stds[c];
			}
		}
	}
	array<int64_t, 4> input_shape_{ 1, dstimg.channels(), m_inpHeight, m_inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, m_vInputImageData.data(), m_vInputImageData.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = m_ortSession->Run(RunOptions{ nullptr }, &m_vInputNames[0], &input_tensor_, 1, m_vOutputNames.data(), m_vOutputNames.size());   // 开始推理
	float* pred = ort_outputs[0].GetTensorMutableData<float>();
	Mat result(m_outHeight, m_outWidth, CV_32FC1, pred);
	//result = 1 - result;
	double min_value, max_value;
	minMaxLoc(result, &min_value, &max_value, 0, 0);
	result = (result - min_value) / (max_value - min_value);
	result *= 255;
	result.convertTo(result, CV_8UC1);
	resize(result, result, Size(srcimg.cols, srcimg.rows));

	return result;
}

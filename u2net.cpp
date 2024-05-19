#include "u2net.h"
#include "CU2Net.h"

int LoadModel(const std::string sModelPath)
{
	return CU2Net::GetInstance()->loadModel(sModelPath);
}

int Detect(const string& sScimgPath, const string& sDstimgPath, const string& sModelPath)
{
	return 0;
}

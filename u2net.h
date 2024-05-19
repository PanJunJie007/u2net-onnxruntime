#pragma once
#include <string>

using namespace std;

#ifdef _IMPORT
#define _DLL_API extern "C" __declspec(dllimport)
#else
#define	_DLL_API extern "C" __declspec(dllexport)
#endif

_DLL_API int LoadModel(const std::string sModelPath);

_DLL_API int Detect(const string& sScimgPath, const string& sDstimgPath, const string& sModelPath);

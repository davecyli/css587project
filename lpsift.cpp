#include "lpsift.h"

LPSIFT::~LPSIFT()
{

}

void LPSIFT::compute(InputArray image, std::vector< KeyPoint >& keypoints, OutputArray descriptors) {
	
};

void LPSIFT::compute(InputArrayOfArrays images, std::vector< std::vector< KeyPoint > >& keypoints, OutputArrayOfArrays descriptors) {

};

int LPSIFT::defaultNorm() const {
	return 0; // temp
};

int LPSIFT::descriptorSize() const {
	return 0; // temp
};

int LPSIFT::descriptorType() const {
	return 0; // temp
};

void LPSIFT::detect(InputArray image, std::vector< KeyPoint >& keypoints, InputArray mask) {};

void LPSIFT::detect(InputArrayOfArrays images, std::vector< std::vector< KeyPoint > >& keypoints, InputArrayOfArrays masks) {};

void LPSIFT::detectAndCompute(InputArray image, InputArray mask, std::vector< KeyPoint >& keypoints, OutputArray descriptors, bool useProvidedKeypoints) {};

bool LPSIFT::empty() const {
	return false; // temp
};

String LPSIFT::getDefaultName() const {
	return "LPSIFT"; // temp
};

void LPSIFT::read(const FileNode&) {};

void LPSIFT::read(const String& fileName) {};

void LPSIFT::write(const Ptr< FileStorage >& fs, const String& name) const {};

void LPSIFT::write(const String& fileName) const {};

void LPSIFT::write(FileStorage&) const {};

void LPSIFT::write(FileStorage& fs, const String& name) const {};
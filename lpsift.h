/*
 * David Li, Ben Schipunov, Kris Yu
 * 11/14/2025
 * CSS 587
 * Final Project: LP-SIFT
 *
 * lpsift.h
 * Header file of LP-SIFT implementation
 * https://arxiv.org/pdf/2405.08578
 *
 * Features included:
 *
 * Assumptions and constraints:
 */

#ifndef LPSIFT_H
#define LPSIFT_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;

class LPSIFT : Feature2D {

private:
public:
	virtual ~LPSIFT();

	virtual void compute(InputArray image, std::vector< KeyPoint >& keypoints, OutputArray descriptors) override;

	virtual void compute(InputArrayOfArrays images, std::vector< std::vector< KeyPoint > >& keypoints, OutputArrayOfArrays descriptors) override;

	virtual int defaultNorm() const override;

	virtual int descriptorSize() const override;

	virtual int descriptorType() const override;

	virtual void detect(InputArray image, std::vector< KeyPoint >& keypoints, InputArray mask = noArray());

	virtual void detect(InputArrayOfArrays images, std::vector< std::vector< KeyPoint > >& keypoints, InputArrayOfArrays masks = noArray());

	virtual void detectAndCompute(InputArray image, InputArray mask, std::vector< KeyPoint >& keypoints, OutputArray descriptors, bool useProvidedKeypoints = false);

	virtual bool empty() const CV_OVERRIDE;

	virtual String getDefaultName() const CV_OVERRIDE;

	virtual void read(const FileNode&) CV_OVERRIDE;

	void read(const String& fileName);

	void write(const Ptr< FileStorage >& fs, const String& name) const;

	void write(const String& fileName) const;

	virtual void write(FileStorage&) const CV_OVERRIDE;

	void write(FileStorage& fs, const String& name) const;

};

#endif
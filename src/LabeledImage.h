#pragma once

#define IMG_SX	28
#define IMG_SY	28

struct LabeledImage
{
	unsigned char	data[IMG_SX*IMG_SY];
	float			floatData[IMG_SX*IMG_SY];
	char			label;

	void save(const char* strFileName) const
	{
		debugSavePGM(data, IMG_SX, IMG_SY, strFileName);
	}
	void updateFloatDataFromData();
};

void readLabeledImages(const char* strImagesFileName, const char* strLabelsFileName, std::vector<LabeledImage>& labeledImages);

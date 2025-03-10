#include "LabeledImage.h"

static unsigned int _readU32(const unsigned char*& curDataPtr)
{
	U32Union u;
	////memcpy(u.u8, &buffer[i], 4);
	//u.u8[0] = buffer[curIndex+3];
	//u.u8[1] = buffer[curIndex+2];
	//u.u8[2] = buffer[curIndex+1];
	//u.u8[3] = buffer[curIndex+0];
	//curIndex += 4;

	u.u8[0] = curDataPtr[3];
	u.u8[1] = curDataPtr[2];
	u.u8[2] = curDataPtr[1];
	u.u8[3] = curDataPtr[0];
	curDataPtr += 4;
	return u.u32;
}

static void _readFileData(const char* strFileName, std::vector<unsigned char>& buffer)
{
	FILE* f = fopen(strFileName, "rb");
	assert(f);

	fseek(f, 0, SEEK_END);
	const long fileSize = ftell(f);
	fseek(f, 0, SEEK_SET);

	buffer.resize(fileSize);
	fread(buffer.data(), 1, fileSize, f);
	
	fclose(f);
}

static void _readImages(const char* strFileName, std::vector<LabeledImage>& labeledImages)
{
	std::vector<unsigned char> buffer;
	_readFileData(strFileName, buffer);
	const unsigned char* curDataPtr = buffer.data();
	
	// ===== IMAGES FILE FORMAT =====
	//[offset] [type]          [value]          [description]
	//0000     32 bit integer  0x00000803(2051) magic number
	//0004     32 bit integer  60000            number of images
	//0008     32 bit integer  28               number of rows
	//0012     32 bit integer  28               number of columns
	//0016     unsigned byte   ??               pixel
	//0017     unsigned byte   ??               pixel
	//........
	//xxxx     unsigned byte   ??               pixel
	unsigned int magic		= _readU32(curDataPtr);
	unsigned int nbImages	= _readU32(curDataPtr);
	unsigned int nbRows		= _readU32(curDataPtr);	assert(nbRows == IMG_SY);
	unsigned int nbCols		= _readU32(curDataPtr);	assert(nbCols == IMG_SX);

	labeledImages.resize(nbImages);

	// "Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black)."
	for(unsigned int idxCurImage=0 ; idxCurImage < nbImages ; idxCurImage++)
	{
		LabeledImage& img = labeledImages[idxCurImage];
		for(int y=0 ; y < IMG_SY ; y++)
		{
			for(int x=0 ; x < IMG_SX ; x++)
			{
				img.data[x + y*IMG_SX] = *curDataPtr;
				curDataPtr++;
			}
		}

		img.updateFloatDataFromData();
	}
}

void LabeledImage::updateFloatDataFromData()
{
	for(int y=0 ; y < IMG_SY ; y++)
	{
		for(int x=0 ; x < IMG_SX ; x++)
		{
			floatData[x + y*IMG_SX] = ((float)data[x + y*IMG_SX]) / 255.f;
		}
	}
}

static void _readLabels(const char* strFileName, std::vector<LabeledImage>& labeledImages)
{
	// ===== LABELS FILE FORMAT =====
	//[offset] [type]          [value]          [description]
	//0000     32 bit integer  0x00000801(2049) magic number (MSB first)
	//0004     32 bit integer  10000            number of items
	//0008     unsigned byte   ??               label
	//0009     unsigned byte   ??               label
	//........
	//xxxx     unsigned byte   ??               label

	std::vector<unsigned char> buffer;
	_readFileData(strFileName, buffer);
	const unsigned char* curDataPtr = buffer.data();

	unsigned int magic		= _readU32(curDataPtr);
	unsigned int nbItems	= _readU32(curDataPtr);
	assert(nbItems == labeledImages.size());

	for(unsigned idxCurItem=0 ; idxCurItem < nbItems ; idxCurItem++)
	{
		labeledImages[idxCurItem].label = *curDataPtr++;
	}
}

void readLabeledImages(const char* strImagesFileName, const char* strLabelsFileName, std::vector<LabeledImage>& labeledImages)
{
	printf("Reading images from %s ...\n", strImagesFileName);
	_readImages(strImagesFileName, labeledImages);

	printf("Reading labels from %s ...\n", strLabelsFileName);
	_readLabels(strLabelsFileName, labeledImages);
}

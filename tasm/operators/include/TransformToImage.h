#ifndef TASM_TRANSFORMTOIMAGE_H
#define TASM_TRANSFORMTOIMAGE_H

#include "Operator.h"

#include "DecodedPixelData.h"
#include "ImageUtilities.h"

namespace tasm {

class TransformToImage : public Operator<std::unique_ptr<std::vector<ImagePtr>>> {
public:
    TransformToImage(std::shared_ptr<Operator<GPUPixelDataContainer>> parent,
            unsigned int maxWidth,
            unsigned int maxHeight)
            : parent_(parent),
            maxWidth_(maxWidth),
            maxHeight_(maxHeight),
            isComplete_(false),
            tmpImage_(0),
            tmpImagePitch_(numChannels_ * maxWidth_) {}

    ~TransformToImage() override {
        if (!tmpImage_)
            return;

        auto result = cuMemFree(tmpImage_);
        assert(result == CUDA_SUCCESS);
    }

    bool isComplete() override { return isComplete_; }
    std::optional<std::unique_ptr<std::vector<ImagePtr>>> next() override;

private:
    std::shared_ptr<Operator<GPUPixelDataContainer>> parent_;
    unsigned int maxWidth_;
    unsigned int maxHeight_;
    bool isComplete_;

    CUdeviceptr tmpImage_;
    static const unsigned int numChannels_ = 4;
    unsigned int tmpImagePitch_;
};

} // namespace tasm

#endif //TASM_TRANSFORMTOIMAGE_H
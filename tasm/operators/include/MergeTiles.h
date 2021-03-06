#ifndef TASM_MERGETILES_H
#define TASM_MERGETILES_H

#include "Operator.h"

#include "DecodedPixelData.h"
#include "EncodedData.h"

namespace tasm {
class SemanticDataManager;
class TileLayoutProvider;

class TransformToRGB : public Operator<GPUDecodedFrameData> {
public:
    TransformToRGB(
            std::shared_ptr<ConfigurationOperator<GPUDecodedFrameData>> parent
    )
        : isComplete_(false),
        parent_(parent)
    { }

    bool isComplete() override { return isComplete_; }
    std::optional<GPUDecodedFrameData> next() override;

private:
    bool isComplete_;
    std::shared_ptr<ConfigurationOperator<GPUDecodedFrameData>> parent_;

    static const unsigned int numChannels_ = 4;
};

class MergeTilesOperator : public Operator<GPUPixelDataContainer> {
public:
    MergeTilesOperator(
            std::shared_ptr<Operator<GPUDecodedFrameData>> parent,
            std::shared_ptr<SemanticDataManager> semanticDataManager,
            std::shared_ptr<TileLayoutProvider> tileLayoutProvider)
            : parent_(parent), semanticDataManager_(semanticDataManager),
            tileLayoutProvider_(tileLayoutProvider), isComplete_(false) {}

    bool isComplete() override { return isComplete_; }
    std::optional<GPUPixelDataContainer> next() override;

private:
    std::shared_ptr<Operator<GPUDecodedFrameData>> parent_;
    std::shared_ptr<SemanticDataManager> semanticDataManager_;
    std::shared_ptr<TileLayoutProvider> tileLayoutProvider_;
    bool isComplete_;
};

class TilesToPixelsOperator : public Operator<GPUPixelDataContainer> {
public:
    TilesToPixelsOperator(std::shared_ptr<Operator<GPUDecodedFrameData>> parent)
        : parent_(parent),
        isComplete_(false) {}

    bool isComplete() override { return isComplete_; }
    std::optional<GPUPixelDataContainer> next() override;

private:
    std::shared_ptr<Operator<GPUDecodedFrameData>> parent_;
    std::shared_ptr<TileLayoutProvider> tileLayoutProvider_;
    bool isComplete_;
};

} // namespace tasm

#endif //TASM_MERGETILES_H

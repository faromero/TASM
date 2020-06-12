#ifndef TASM_TILECONFIGURATIONPROVIDER_H
#define TASM_TILECONFIGURATIONPROVIDER_H

#include "TileLayout.h"

namespace tasm {

class TileLayoutProvider {
public:
    virtual const TileLayout &tileLayoutForFrame(unsigned int frame) = 0;
    virtual ~TileLayoutProvider() {}
};

class SingleTileConfigurationProvider: public TileLayoutProvider {
public:
    SingleTileConfigurationProvider(unsigned int totalWidth, unsigned int totalHeight)
            : totalWidth_(totalWidth), totalHeight_(totalHeight), layout_(1, 1, {totalWidth_}, {totalHeight_})
    { }

    const TileLayout &tileLayoutForFrame(unsigned int frame) override {
        return layout_;
    }

private:
    unsigned int totalWidth_;
    unsigned  int totalHeight_;
    TileLayout layout_;
};

class UniformTileconfigurationProvider: public TileLayoutProvider {
public:
    UniformTileconfigurationProvider(unsigned int numRows, unsigned int numColumns, Configuration configuration)
        : numRows_(numRows),
        numColumns_(numColumns),
        configuration_(configuration)
    {}

    const TileLayout &tileLayoutForFrame(unsigned int frame) override {
        if (layoutPtr)
            return *layoutPtr;

        layoutPtr = std::make_unique<TileLayout>(numColumns_, numRows_,
                tile_dimensions(configuration_.codedWidth, configuration_.displayWidth, numColumns_),
                tile_dimensions(configuration_.codedHeight, configuration_.displayHeight, numRows_));
        return *layoutPtr;
    }

private:
    std::vector<unsigned int> tile_dimensions(unsigned int codedDimension, unsigned int displayDimension, unsigned int numTiles) {
//        static unsigned int CTBS_SIZE_Y = 32;
        std::vector<unsigned int> dimensions(numTiles);
        unsigned int total = 0;
        for (auto i = 0u; i < numTiles; ++i) {
            unsigned int proposedDimension = ((i + 1) * codedDimension / numTiles) - (i * codedDimension / numTiles);
            if (total + proposedDimension > displayDimension)
                proposedDimension = displayDimension - total;

            dimensions[i] = proposedDimension;
            total += proposedDimension;
        }
        return dimensions;
    }

    unsigned int numRows_;
    unsigned int numColumns_;
    Configuration configuration_;
    std::unique_ptr<TileLayout> layoutPtr;
};

} // namespace tasm

#endif //TASM_TILECONFIGURATIONPROVIDER_H

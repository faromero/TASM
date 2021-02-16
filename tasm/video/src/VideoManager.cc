#include "VideoManager.h"

#include "ImageUtilities.h"
#include "MergeTiles.h"
#include "TileLocationProvider.h"
#include "TiledVideoManager.h"
#include "ScanOperators.h"
#include "ScanTiledVideoOperator.h"
#include "DecodeOperators.h"
#include "SemanticIndex.h"
#include "SemanticSelection.h"
#include "SmartTileConfigurationProvider.h"
#include "TemporalSelection.h"
#include "TileOperators.h"
#include "TransformToImage.h"
#include "Video.h"
#include "VideoConfiguration.h"
#include "WorkloadCostEstimator.h"

#include <iostream>
#include <chrono>


namespace tasm {

void VideoManager::createCatalogIfNecessary() {
    if (!std::experimental::filesystem::exists(CatalogConfiguration::CatalogPath()))
        std::experimental::filesystem::create_directory(CatalogConfiguration::CatalogPath());
}

void VideoManager::store(const std::experimental::filesystem::path &path, const std::string &name) {
    auto start = std::chrono::high_resolution_clock::now();

    std::shared_ptr<Video> video(new Video(path));
    auto tileConfigurationProvider = std::make_shared<SingleTileConfigurationProvider>(
            video->configuration().displayWidth,
            video->configuration().displayHeight);
    storeTiledVideo(video, tileConfigurationProvider, name);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();

    std::cout << "VideoManager store time(ms): " << duration << std::endl;
}

void VideoManager::storeWithUniformLayout(const std::experimental::filesystem::path &path, const std::string &name, unsigned int numRows, unsigned int numColumns) {
    auto start = std::chrono::high_resolution_clock::now();

    std::shared_ptr<Video> video(new Video(path));
    auto tileConfigurationProvider = std::make_shared<UniformTileconfigurationProvider>(numRows, numColumns, video->configuration());
    storeTiledVideo(video, tileConfigurationProvider, name);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();

    std::cout << "storeWithUniformLayout time(ms): " << duration << std::endl;
}

void VideoManager::storeWithNonUniformLayout(const std::experimental::filesystem::path &path,
                                                const std::string &storedName,
                                                const std::string &metadataIdentifier,
                                                std::shared_ptr<MetadataSelection> metadataSelection,
                                                std::shared_ptr<SemanticIndex> semanticIndex, bool force) {
    auto start = std::chrono::high_resolution_clock::now();
    std::shared_ptr<Video> video(new Video(path));
    auto semanticDataManager = std::make_shared<SemanticDataManager>(semanticIndex, metadataIdentifier, metadataSelection, std::shared_ptr<TemporalSelection>());
    std::shared_ptr<TileLayoutProvider> layoutProvider;

    auto layoutDuration = video->configuration().frameRate;
    auto width = video->configuration().displayWidth;
    auto height = video->configuration().displayHeight;

    if (force) {
        layoutProvider = std::make_shared<FineGrainedTileConfigurationProvider>(
                layoutDuration,
                semanticDataManager,
                width,
                height);
    } else {
        layoutProvider = std::make_shared<SmartTileConfigurationProviderSingleSelection>(
                layoutDuration,
                semanticDataManager,
                width,
                height);
    }
    storeTiledVideo(video, layoutProvider, storedName);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();

    std::cout << "storeWithNonUniformLayout time(ms): " << duration << std::endl;
}

void VideoManager::storeTiledVideo(std::shared_ptr<Video> video, std::shared_ptr<TileLayoutProvider> tileLayoutProvider, const std::string &savedName) {
    auto start = std::chrono::high_resolution_clock::now();

    std::shared_ptr<ScanFileDecodeReader> scan(new ScanFileDecodeReader(video));
    std::shared_ptr<GPUDecodeFromCPU> decode(new GPUDecodeFromCPU(scan, video->configuration(), gpuContext_, lock_));

    TileOperator tile(video, decode, tileLayoutProvider, savedName, video->configuration().frameRate, gpuContext_, lock_);
    while (!tile.isComplete()) {
        tile.next();
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();

    std::cout << "storeTiledVideo time(ms): " << duration << std::endl;
}

void VideoManager::retileVideoBasedOnRegret(const std::string &videoName) {
    assert(videoToRegretAccumulator_.count(videoName));

    auto start = std::chrono::high_resolution_clock::now();

    auto tiledEntry = std::make_shared<TiledEntry>(videoName);
    auto tiledVideoManager = std::make_shared<TiledVideoManager>(tiledEntry);
    auto video = std::make_shared<Video>(tiledVideoManager->locationOfTileForId(0, 0));
    auto gopLength = video->configuration().frameRate;

    auto gopToLayouts = videoToRegretAccumulator_.at(videoName)->getNewGOPLayouts();
    // Because we re-tile the entire GOP, we only need to specify the first frame for each GOP.
    auto frames = std::make_shared<std::vector<int>>();
    for (auto it = gopToLayouts->begin(); it != gopToLayouts->end(); ++it)
        frames->push_back(it->first * gopLength);

    // Sort the frames because currently the way we scan goes in order of keyframes.
    // That should probably get more flexible, but for now sorting is easy.
    std::sort(frames->begin(), frames->end());

    retileVideo(video, frames, std::make_shared<ConglomerationTileConfigurationProvider>(std::move(gopToLayouts), gopLength), videoName);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();

    std::cout << "retileBasedOnRegret time(ms): " << duration << std::endl;
}

void VideoManager::retileVideo(std::shared_ptr<Video> video, std::shared_ptr<std::vector<int>> framesToRead, std::shared_ptr<TileLayoutProvider> newLayoutProvider, const std::string &savedName) {
    // Set up scan of original video using specified frames. Re-tile entire GOPs, even if not every frame is specified.
    auto scan = std::make_shared<ScanFramesFromFileDecodeReader>(video, framesToRead, true);
    auto decode = std::make_shared<GPUDecodeFromCPU>(scan, video->configuration(), gpuContext_, lock_);

    TileOperator tile(video, decode, newLayoutProvider, savedName, video->configuration().frameRate, gpuContext_, lock_);
    while (!tile.isComplete()) {
        tile.next();
    }
}

std::unique_ptr<ImageIterator> VideoManager::select(const std::string &video,
                                                    const std::string &metadataIdentifier,
                                                    std::shared_ptr<MetadataSelection> metadataSelection,
                                                    std::shared_ptr<TemporalSelection> temporalSelection,
                                                    std::shared_ptr<SemanticIndex> semanticIndex,
                                                    SelectStrategy selectStrategy) {
    auto start = std::chrono::high_resolution_clock::now();

    std::shared_ptr<TiledEntry> entry(new TiledEntry(video, metadataIdentifier));

    // Set up scan of a tiled video.
    std::shared_ptr<TiledVideoManager> tiledVideoManager(new TiledVideoManager(entry));
    auto tileLocationProvider = std::make_shared<SingleTileLocationProvider>(tiledVideoManager);
    auto semanticDataManager = std::make_shared<SemanticDataManager>(semanticIndex, metadataIdentifier, metadataSelection, temporalSelection, tiledVideoManager->totalWidth(), tiledVideoManager->totalHeight());

    std::shared_ptr<Operator<CPUEncodedFrameDataPtr>> scan;
    std::shared_ptr<TileLayoutProvider> tileLayoutProvider = tileLocationProvider;

    // Set up default configuration info.
    auto maxWidth = tiledVideoManager->largestWidth();
    auto maxHeight = tiledVideoManager->largestHeight();
    // Specify largest tile dimensions which are required to successfully reconfigure the decoder.
    // The maximum dimensions were set based on display dimensions; make sure they are big enough to handle larger coded dimensions.
    static const unsigned int CodedDimension = 32;
    if (maxWidth % CodedDimension)
        maxWidth = (maxWidth / CodedDimension + 1) * CodedDimension;
    if (maxHeight % CodedDimension)
        maxHeight = (maxHeight / CodedDimension + 1) * CodedDimension;

    auto configuration = *video::GetConfiguration(tileLocationProvider->locationOfTileForFrame(0, 0));
    configuration.maxWidth = maxWidth;
    configuration.maxHeight = maxHeight;

    if (selectStrategy == SelectStrategy::Frames) {
        auto scanFullFrames = std::make_shared<ScanFullFramesFromTiledVideoOperator>(entry, semanticDataManager, tileLocationProvider);
        scan = scanFullFrames;

        // Create a layout provider for the full frame.
        auto layout = tileLocationProvider->tileLayoutForFrame(0);
        tileLayoutProvider = std::make_shared<SingleTileConfigurationProvider>(layout->totalWidth(), layout->totalHeight());

        // Use the full frame configuration.
        configuration = scanFullFrames->configuration();
        maxWidth = configuration.maxWidth;
        maxHeight = configuration.maxHeight;
    } else {
        scan = std::make_shared<ScanTiledVideoOperator>(entry, semanticDataManager, tileLocationProvider);
    }

    std::shared_ptr<GPUDecodeFromCPU> decode(new GPUDecodeFromCPU(scan, configuration, gpuContext_, lock_, maxWidth, maxHeight));
    auto toRGB = std::make_shared<TransformToRGB>(decode);

    // Transform tiles to pixel blobs.
    std::shared_ptr<Operator<GPUPixelDataContainer>> mergeOperator;
    if (selectStrategy == SelectStrategy::Objects) {
        std::cout << "Merging pixels to recover objects" << std::endl;
        mergeOperator = std::make_shared<MergeTilesOperator>(toRGB, semanticDataManager, tileLayoutProvider);
    } else {
        std::cout << "Returning raw tiles" << std::endl;
        mergeOperator = std::make_shared<TilesToPixelsOperator>(toRGB);
    }

    // Transform pixels to RGB images.
    std::shared_ptr<TransformToImage> transform(new TransformToImage(mergeOperator, maxWidth, maxHeight));

    // Accumulate regret for this query.
    if (videoToRegretAccumulator_.count(video))
        accumulateRegret(video, semanticDataManager, tileLocationProvider);

    return std::make_unique<ImageIterator>(transform);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();

    std::cout << "VideoManager select time(ms): " << duration << std::endl;
}

void VideoManager::accumulateRegret(const std::string &video, std::shared_ptr<SemanticDataManager> selection, std::shared_ptr<TileLayoutProvider> currentLayout) {
    auto regretAccumulator = videoToRegretAccumulator_.at(video);

    // Create a workload.
    auto workload = std::make_shared<Workload>(selection);

    // Add regret for this query and get GOPs that have accumulated enough regret to be re-tiled.
    regretAccumulator->addRegretForQuery(workload, currentLayout);
}

void VideoManager::activateRegretBasedRetilingForVideo(const std::string &video, const std::string &metadataIdentifier, std::shared_ptr<SemanticIndex> semanticIndex, double threshold) {
    std::shared_ptr<TiledEntry> entry(new TiledEntry(video, metadataIdentifier));
    std::shared_ptr<TiledVideoManager> tiledVideoManager(new TiledVideoManager(entry));
    Video originalVideo(tiledVideoManager->locationOfTileForId(0, 0));

    videoToRegretAccumulator_[video] = std::make_shared<RegretAccumulator>(
            semanticIndex,
            metadataIdentifier,
            tiledVideoManager->totalWidth(),
            tiledVideoManager->totalHeight(),
            originalVideo.configuration().frameRate,
            threshold);
}

void VideoManager::deactivateRegretBasedRetilingForVideo(const std::string &video) {
    videoToRegretAccumulator_.erase(video);
}

} // namespace tasm

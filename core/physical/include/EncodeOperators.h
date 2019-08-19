#ifndef LIGHTDB_ENCODEOPERATORS_H
#define LIGHTDB_ENCODEOPERATORS_H

#include "LightField.h"
#include "PhysicalOperators.h"
#include "Codec.h"
#include "Configuration.h"
#include "VideoEncoder.h"
#include "EncodeWriter.h"
#include "VideoEncoderSession.h"
#include "MaterializedLightField.h"
#include <cstdio>
#include <utility>

#include "timer.h"

namespace lightdb::physical {

class GPUEncodeToCPU : public PhysicalOperator, public GPUOperator {
public:
    static constexpr size_t kDefaultGopSize = 30;

    explicit GPUEncodeToCPU(const LightFieldReference &logical,
                            PhysicalOperatorReference &parent,
                            Codec codec)
            : PhysicalOperator(logical, {parent}, DeviceType::GPU, runtime::make<Runtime>(*this, "GPUEncodeToCPU-init")),
              GPUOperator(parent),
              codec_(std::move(codec)),
              didSetFramesToKeep_(false) {
        if(!codec.nvidiaId().has_value())
            throw GpuRuntimeError("Requested codec does not have an Nvidia encode id");
    }

    const Codec &codec() const { return codec_; }

    bool didSetFramesToKeep() const { return didSetFramesToKeep_; }
    const std::set<int> &framesToKeep() const { return framesToKeep_; }
    void setFramesToKeep(const std::unordered_set<int> &framesToKeep) {
        didSetFramesToKeep_ = true;
        framesToKeep_.insert(framesToKeep.begin(), framesToKeep.end());
    }

    void setFramesToKeep(const std::vector<int> &framesToKeep) {
        didSetFramesToKeep_ = true;
        framesToKeep_.insert(framesToKeep.begin(), framesToKeep.end());
    }

    bool didSetDesiredKeyframes() const { return didSetDesiredKeyframes_; }
    const std::unordered_set<int> &desiredKeyframes() const { return desiredKeyframes_; }
    void setDesiredKeyframes(std::unordered_set<int> keyframes) {
        didSetDesiredKeyframes_ = true;
        desiredKeyframes_ = std::move(keyframes);
    }

private:
    class Runtime: public runtime::GPUUnaryRuntime<GPUEncodeToCPU, GPUDecodedFrameData> {
    public:
        explicit Runtime(GPUEncodeToCPU &physical)
            : runtime::GPUUnaryRuntime<GPUEncodeToCPU, GPUDecodedFrameData>(physical),
              encodeConfiguration_{configuration(), this->physical().codec().nvidiaId().value(), gop()},
              encoder_{this->context(), encodeConfiguration_, lock()},
              writer_{encoder_.api()},
              encodeSession_{encoder_, writer_},
              numberOfEncodedFrames_(0),
              frameNumber_(0),
              nextFrameThatWillBeEncoded_(physical.framesToKeep().begin()),
              currentDataFramesIteratorIsValid_(false)
        {
            desiredKeyframes_ = physical.didSetDesiredKeyframes() ? physical.desiredKeyframes() : getDesiredKeyframes();
        }

        std::optional<physical::MaterializedLightFieldReference> read() override {
            // TODO: We only want to pass on 1 GOP at a time (when selecting frames with multiple strategies).
            // Should this keep track of GOPs instead of encoding frames from many GOPs at a time?
            // Would that cause the writer to dequeue frames from 1 GOP at a time?
            GLOBAL_TIMER.startSection("GPUEncodeToCPU");
            if (iterator() != iterator().eos()) {
                auto &decoded = *iterator();
                if (!currentDataFramesIteratorIsValid_) {
                    currentDataFramesIterator_ = decoded.frames().begin();
                    currentDataFramesIteratorIsValid_ = true;
                }

                timer_.startSection("inside-GPUEncodeToCPU");
                bool encodedKeyframe = false;
                for (auto &frameIt = currentDataFramesIterator_; frameIt != decoded.frames().end(); frameIt++) {
                    if (encodedKeyframe)
                        break;

                    int frameNumberFromFrame = -1;
                    auto frameNumber = (*frameIt)->getFrameNumber(frameNumberFromFrame) ? frameNumberFromFrame : frameNumber_;
                    ++frameNumber_; // Always increment this so we can keep track of how many frames were decoded.
                    if (physical().didSetFramesToKeep() && !physical().framesToKeep().count(frameNumber))
                        continue;

                    ++numberOfEncodedFrames_;
                    // Try to force the output of 1 GOP at a time.
                    if (desiredKeyframes_.count(frameNumber)) {
                        // Flush the last GOP.
                        encodeSession_.Flush();
                        encodedKeyframe = true;
                    }

                    encodeSession_.Encode(**frameIt, decoded.configuration().offset.top,
                                          decoded.configuration().offset.left,
                                          desiredKeyframes_.count(frameNumber));
                }

                auto configuration = decoded.configuration();
                auto geometry = decoded.geometry();

                if (noMoreFramesInCurrentData()) {
                    ++iterator();
                    currentDataFramesIteratorIsValid_ = false;
                }

                if (iterator() == iterator().eos())
                    // If so, flush the encode queue and end this op too
                    encodeSession_.Flush();

                // For mixed frame selection, we need to export this one GOP at a time.
                unsigned int numberOfFrames = 0;
                auto returnVal = CPUEncodedFrameData(physical().codec(), configuration, geometry, writer_.dequeue(numberOfFrames));
                returnVal.setFirstFrameIndexAndNumberOfFrames(*nextFrameThatWillBeEncoded_, numberOfFrames);
                std::advance(nextFrameThatWillBeEncoded_, numberOfFrames);

                GLOBAL_TIMER.endSection("GPUEncodeToCPU");
                timer_.endSection("inside-GPUEncodeToCPU");
                return {returnVal};
            } else {
                GLOBAL_TIMER.endSection("GPUEncodeToCPU");
                std::cout << "**Encoded " << numberOfEncodedFrames_ << " frames, considered " << frameNumber_ << "frames\n";
                timer_.printAllTimes();
                return std::nullopt;
            }
        }

    private:
        unsigned int gop() const {
            auto option = logical().is<OptionContainer<>>()
                          ? logical().downcast<OptionContainer<>>().get_option(EncodeOptions::GOPSize)
                          : std::nullopt;

            if(option.has_value() && option.value().type() != typeid(unsigned int))
                throw InvalidArgumentError("Invalid GOP option specified", EncodeOptions::GOPSize);
            else
                return std::any_cast<unsigned int>(option.value_or(
                        std::make_any<unsigned int>(kDefaultGopSize)));
        }

        std::unordered_set<int> getDesiredKeyframes() const {
            auto option = logical().is<OptionContainer<>>()
                    ? logical().downcast<OptionContainer<>>().get_option(EncodeOptions::Keyframes)
                    : std::nullopt;

            if (option.has_value() && option.value().type() != typeid(std::unordered_set<int>))
                throw InvalidArgumentError("Invalid keyframes option specified", EncodeOptions::Keyframes);
            else if (option.has_value())
                return std::any_cast<std::unordered_set<int>>(option.value());
            else
                return {};
        }

        bool noMoreFramesInCurrentData() {
            return currentDataFramesIterator_ == (*iterator()).frames().end();
        }

        EncodeConfiguration encodeConfiguration_;
        VideoEncoder encoder_;
        MemoryEncodeWriter writer_;
        VideoEncoderSession encodeSession_;
        Timer timer_;
        unsigned long numberOfEncodedFrames_;
        unsigned long frameNumber_;
        std::unordered_set<int> desiredKeyframes_;
        std::set<int>::const_iterator nextFrameThatWillBeEncoded_;
        std::vector<GPUFrameReference>::const_iterator currentDataFramesIterator_;
        bool currentDataFramesIteratorIsValid_;
    };

    const Codec codec_;

    // Add optional list of frames to keep.
    bool didSetFramesToKeep_;
    std::set<int> framesToKeep_;

    bool didSetDesiredKeyframes_;
    std::unordered_set<int> desiredKeyframes_;
};

}; // namespace lightdb::physical

#endif //LIGHTDB_ENCODEOPERATORS_H

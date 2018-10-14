#ifndef LIGHTDB_STOREOPERATORS_H
#define LIGHTDB_STOREOPERATORS_H

#include "PhysicalOperators.h"

namespace lightdb::physical {

class Store: public PhysicalLightField, public EncodedVideoInterface  {
public:
    explicit Store(const LightFieldReference &logical,
                   PhysicalLightFieldReference &parent)
            : Store(logical,
                    logical->downcast<logical::StoredLightField>(),
                    parent)
    { }

    std::optional<physical::MaterializedLightFieldReference> read() override {
        if(!all_parent_eos()) {
            auto data = iterators()[0]++;
            auto &encoded = data.downcast<physical::CPUEncodedFrameData>();
            std::copy(encoded.value().begin(),encoded.value().end(), std::ostreambuf_iterator<char>(output_->stream()));
            return data;
        } else
            return std::nullopt;
    }

    const Codec &codec() const override { return store_.codec(); }
    const Configuration &configuration() override { return configuration_.value(); }

private:
    explicit Store(const LightFieldReference &logical,
                   const logical::StoredLightField &store,
                   PhysicalLightFieldReference &parent)
            : PhysicalLightField(logical, {parent}, DeviceType::CPU),
              store_(store),
              parent_(parent),
              configuration_([this]() { return parent_.downcast<EncodedVideoInterface>().configuration(); }),
              output_([this]() { return catalog::Catalog::instance().create(store_.name(), parent_.downcast<EncodedVideoInterface>().codec(), configuration_); }) {
        LOG(INFO) << "Storing to ambient catalog";
        CHECK_EQ(parents().size(), 1);
    }

    const logical::StoredLightField &store_;
    PhysicalLightFieldReference parent_;
    lazy<Configuration> configuration_;
    lazy<catalog::OutputStream> output_;
};

} // namespace lightdb::physical

#endif //LIGHTDB_STOREOPERATORS_H

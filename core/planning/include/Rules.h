#ifndef LIGHTDB_RULES_H
#define LIGHTDB_RULES_H

#include "Model.h"
#include "Optimizer.h"
#include "ScanOperators.h"
#include "EncodeOperators.h"
#include "DecodeOperators.h"
#include "UnionOperators.h"
#include "MapOperators.h"
#include "TransferOperators.h"
#include "DiscretizeOperators.h"
#include "InterpolateOperators.h"
#include "SubqueryOperators.h"
#include "IdentityOperators.h"
#include "HomomorphicOperators.h"
#include "SubsetOperators.h"
#include "StoreOperators.h"
#include "SinkOperators.h"
#include "Rectangle.h"

namespace lightdb::optimization {
    class ChooseMaterializedScans : public OptimizerRule {
    public:
        using OptimizerRule::OptimizerRule;

        bool visit(const LightField &node) override {
            if (plan().has_physical_assignment(node))
                return false;
            else if(node.is<physical::GPUDecodedFrameData>()) {
                auto ref = plan().lookup(node);
                auto &m = node.downcast<physical::GPUDecodedFrameData>();
                auto mref = physical::MaterializedLightFieldReference::make<physical::GPUDecodedFrameData>(m);

                //Removed this line without actually testing the consequences
                //plan().emplace<physical::GPUScanMemory>(ref, mref);
                //plan().emplace<physical::GPUOperatorAdapter>(mref);
                // Made this change without testing it -- when is this rule fired?
                plan().emplace<physical::MaterializedToPhysicalOperatorAdapter>(ref, mref);
                return true;
            } else if(node.is<physical::PhysicalToLogicalLightFieldAdapter>()) {
                auto ref = plan().lookup(node);
                auto op = plan().emplace<physical::GPUOperatorAdapter>(ref.downcast<physical::PhysicalToLogicalLightFieldAdapter>().source());
                plan().assign(ref, op);
                return true;
            } else
                return false;
        }
    };

    class ChooseDecoders : public OptimizerRule {
    public:
        using OptimizerRule::OptimizerRule;

        bool visit(const logical::ScannedLightField &node) override {
            if(!plan().has_physical_assignment(node)) {
                LOG(WARNING) << "Just using first stream and GPU and ignoring all others";
                auto stream = node.metadata().streams()[0];
                auto logical = plan().lookup(node);

                if(stream.codec() == Codec::h264() ||
                   stream.codec() == Codec::hevc()) {
                    auto gpu = plan().environment().gpus()[0];

                    auto &scan = plan().emplace<physical::ScanSingleFileToGPU>(logical, stream);
                    auto decode = plan().emplace<physical::GPUDecode>(logical, scan, gpu);

                    auto children = plan().children(plan().lookup(node));
                    if(children.size() > 1) {
                        auto tees = physical::TeedPhysicalLightFieldAdapter::make(decode, children.size());
                        for (auto index = 0u; index < children.size(); index++)
                            plan().emplace<physical::GPUOperatorAdapter>(tees->physical(index), decode);
                    }
                } else if(stream.codec() == Codec::boxes()) {
                    auto &scan = plan().emplace<physical::ScanSingleFile<sizeof(Rectangle) * 8192>>(logical, stream);
                    auto decode = plan().emplace<physical::CPUFixedLengthRecordDecode<Rectangle>>(logical, scan);
                }

                return true;
            }
            return false;
        }
    };

    class ChooseEncoders : public OptimizerRule {
    public:
        using OptimizerRule::OptimizerRule;

        bool visit(const logical::EncodedLightField &node) override {
            if(!plan().has_physical_assignment(node)) {
                auto physical_parents = functional::flatmap<std::vector<PhysicalLightFieldReference>>(
                        node.parents().begin(), node.parents().end(),
                        [this](auto &parent) { return plan().unassigned(parent); });

                if(physical_parents.empty())
                    return false;

                //TODO clean this up, shouldn't just be randomly picking last parent
                auto physical_parent = physical_parents[0];
                auto logical = plan().lookup(node);

                if(physical_parent.is<physical::GPUAngularSubquery>() && physical_parent.downcast<physical::GPUAngularSubquery>().subqueryType().is<logical::EncodedLightField>())
                    plan().emplace<physical::CPUIdentity>(logical, physical_parent);
                else if(physical_parent.is<physical::GPUOperator>() && node.codec().nvidiaId().has_value())
                    plan().emplace<physical::GPUEncode>(logical, physical_parent, node.codec());
                else if(physical_parent.is<physical::GPUOperator>() && node.codec() == Codec::raw())
                    plan().emplace<physical::GPUEnsureFrameCropped>(plan().lookup(node), physical_parent);
                else if(physical_parent.is<physical::CPUMap>() && physical_parent.downcast<physical::CPUMap>().transform()(physical::DeviceType::CPU).codec().name() == node.codec().name())
                    plan().emplace<physical::CPUIdentity>(logical, physical_parent);
                //TODO this is silly -- every physical operator should declare an output type and we should just use that
                else if(physical_parent.is<physical::TeedPhysicalLightFieldAdapter::TeedPhysicalLightField>() && physical_parent->parents()[0].is<physical::CPUMap>() && physical_parent->parents()[0].downcast<physical::CPUMap>().transform()(physical::DeviceType::CPU).codec().name() == node.codec().name())
                    plan().emplace<physical::CPUIdentity>(logical, physical_parent);
                else if(physical_parent->device() == physical::DeviceType::CPU) {
                    auto gpu = plan().environment().gpus()[0];
                    auto transfer = plan().emplace<physical::CPUtoGPUTransfer>(logical, physical_parent, gpu);
                    plan().emplace<physical::GPUEncode>(plan().lookup(node), transfer, node.codec());
                } else
                    return false;
                return true;
            } else {

            }
            return false;
        }
    };

    class ChooseUnion : public OptimizerRule {
    public:
        using OptimizerRule::OptimizerRule;

        std::optional<PhysicalLightFieldReference> FindEncodeParent(const LightFieldReference &node) {
            for(const auto &parent: node->parents()) {
                if(parent.is<logical::EncodedLightField>() &&
                   plan().has_physical_assignment(parent))
                    return {plan().assignments(parent)[0]};
            }
            return {};
        }

        std::optional<PhysicalLightFieldReference> FindHomomorphicAncestor(const LightFieldReference &node) {
            std::deque<PhysicalLightFieldReference> queue;

            for(const auto &parent: node->parents())
                for(const auto &assignment: plan().assignments(parent))
                    queue.push_back(assignment);

            while(!queue.empty()) {
                auto element = queue.front();
                queue.pop_front();

                if(element->logical().is<logical::CompositeLightField>() &&
                   element.is<physical::HomomorphicUniformAngularUnion>())
                    return element;
                else if(element.is<physical::CPUIdentity>())
                    for(const auto &parent: element->parents())
                        queue.push_back(parent);
            }

            return {};
        }

        bool TryGPUBoxOverlayUnion(const logical::CompositeLightField &node) {
            auto leafs0 = plan().unassigned(node.parents()[0]);
            auto leafs1 = plan().unassigned(node.parents()[1]);
            if(leafs0.size() != 1 || leafs1.size() != 1)
                return false;
            //TODO shouldn't arbitrarily require a shallow union
            else if(!node.parents()[0].is<logical::ScannedLightField>() || !node.parents()[1].is<logical::ScannedLightField>())
                return false;
            //TODO should pay attention to all streams
            else if(node.parents()[0].downcast<logical::ScannedLightField>().metadata().streams()[0].codec() != Codec::boxes())
                return false;
            else if(node.parents()[1].downcast<logical::ScannedLightField>().metadata().streams()[0].codec() != Codec::h264() &&
                    node.parents()[1].downcast<logical::ScannedLightField>().metadata().streams()[0].codec() != Codec::hevc())
                return false;
            else {
                auto unioned = plan().emplace<physical::GPUBoxOverlayUnion>(plan().lookup(node), std::vector<PhysicalLightFieldReference>{leafs0[0], leafs1[0]});

                auto children = plan().children(plan().lookup(node));
                if(children.size() > 1) {
                    auto tees = physical::TeedPhysicalLightFieldAdapter::make(unioned, children.size());
                    for (auto index = 0u; index < children.size(); index++) {
                        if (unioned->device() == physical::DeviceType::CPU)
                            plan().assign(plan().lookup(node), tees->physical(index));
                        else if (unioned->device() == physical::DeviceType::GPU)
                            plan().emplace<physical::GPUOperatorAdapter>(tees->physical(index), unioned);
                        else
                            throw InvalidArgumentError("No rule support for device type.", "node");
                    }
                }
                
                return true;
            }
        }

        bool visit(const logical::CompositeLightField &node) override {
            if (plan().has_physical_assignment(node))
                return false;
            else if (node.parents().size() != 2)
                return false;
            else if (node.parents()[0].is<logical::EncodedLightField>() &&
                     node.parents()[1].is<logical::EncodedLightField>()) {
                std::vector<PhysicalLightFieldReference> physical_outputs;
                for (auto &parent: node.parents()) {
                    const auto &assignments = plan().assignments(parent);
                    if (assignments.empty())
                        return false;
                    physical_outputs.push_back(assignments[assignments.size() - 1]);
                }

                plan().emplace<physical::HomomorphicUniformAngularUnion>(plan().lookup(node), physical_outputs, 4, 4);
                return true;
            } else if (FindHomomorphicAncestor(node).has_value() &&
                       FindEncodeParent(node).has_value()) {
                auto href = FindHomomorphicAncestor(node).value();
                auto eref = FindEncodeParent(node).value();

                href->parents().emplace_back(eref);
                plan().assign(node, href);
                plan().assign(node, eref);

                plan().emplace<physical::CPUIdentity>(plan().lookup(node), href);
                return true;
            } else if(TryGPUBoxOverlayUnion(node)) {
                return true;
            } else
                return false;
        }
    };

    class ChooseMap : public OptimizerRule {
    public:
        using OptimizerRule::OptimizerRule;

        PhysicalLightFieldReference Map(const logical::TransformedLightField &node, PhysicalLightFieldReference parent) {
            auto logical = plan().lookup(node);

            if(!node.functor()->has_implementation(physical::DeviceType::GPU)) {
                auto transfer = plan().emplace<physical::GPUtoCPUTransfer>(plan().lookup(node), parent);
                return plan().emplace<physical::CPUMap>(plan().lookup(node), transfer, *node.functor());
            } else
                return plan().emplace<physical::GPUMap>(plan().lookup(node), parent, *node.functor());
        }


        bool visit(const logical::TransformedLightField &node) override {
            if(!plan().has_physical_assignment(node)) {
                auto physical_parents = functional::flatmap<std::vector<PhysicalLightFieldReference>>(
                        node.parents().begin(), node.parents().end(),
                        [this](auto &parent) { return plan().unassigned(parent); });

                if(!physical_parents.empty()) {
                    auto mapped = Map(node, physical_parents[0]);

                    //TODO what if function isn't determistic?!

                    auto children = plan().children(plan().lookup(node));
                    if(children.size() > 1) {
                        auto tees = physical::TeedPhysicalLightFieldAdapter::make(mapped, children.size());
                        for(auto index = 0u; index < children.size(); index++) {
                            if(mapped->device() == physical::DeviceType::CPU)
                                plan().assign(plan().lookup(node), tees->physical(index));
                            else if(mapped->device() == physical::DeviceType::GPU)
                                plan().emplace<physical::GPUOperatorAdapter>(tees->physical(index), mapped);
                            else
                                throw InvalidArgumentError("No rule support for device type.", "node");
                        }
                    }
                    return true;
                }
            }
            return false;
        }
    };

    class ChooseSelection : public OptimizerRule {
    public:
        using OptimizerRule::OptimizerRule;

        PhysicalLightFieldReference AngularSelection(const logical::SubsetLightField &node,
                                                     PhysicalLightFieldReference parent) {
            if(parent->device() == physical::DeviceType::GPU)
                return plan().emplace<physical::GPUAngularSubframe>(plan().lookup(node), parent);
            else
                throw std::runtime_error("Hardcoded support only for GPU angular selection"); //TODO
        }

        PhysicalLightFieldReference TemporalSelection(const logical::SubsetLightField &node,
                                                      PhysicalLightFieldReference parent) {
            LOG(WARNING) << "Assuming temporal selection parent is encoded video without actually checking";
            if(parent->device() == physical::DeviceType::GPU)
                return plan().emplace<physical::FrameSubset>(plan().lookup(node), parent);
            else
                throw std::runtime_error("Hardcoded support only for GPU temporal selection"); //TODO
        }

        PhysicalLightFieldReference IdentitySelection(const logical::SubsetLightField &node,
                                                      PhysicalLightFieldReference parent) {
            if(parent->device() == physical::DeviceType::CPU)
                return plan().emplace<physical::CPUIdentity>(plan().lookup(node), parent);
            else if(parent->device() == physical::DeviceType::GPU)
                return plan().emplace<physical::GPUIdentity>(plan().lookup(node), parent);
            else
                throw std::runtime_error("No identity support for FPGA"); //TODO
        }

        bool visit(const logical::SubsetLightField &node) override {
            if(!plan().has_physical_assignment(node)) {
                auto physical_parents = functional::flatmap<std::vector<PhysicalLightFieldReference>>(
                        node.parents().begin(), node.parents().end(),
                        [this](auto &parent) { return plan().unassigned(parent); });

                if(physical_parents.empty())
                    return false;

                auto selection = physical_parents[0];
                auto dimensions = node.dimensions();

                selection = dimensions.find(Dimension::Theta) != dimensions.end() ||
                            dimensions.find(Dimension::Phi) != dimensions.end()
                    ? AngularSelection(node, selection)
                    : selection;

                selection = dimensions.find(Dimension::Time) != dimensions.end()
                    ? TemporalSelection(node, selection)
                    : selection;

                selection = dimensions.empty()
                    ? IdentitySelection(node, selection)
                    : selection;

                if(dimensions.find(Dimension::X) != dimensions.end() ||
                        dimensions.find(Dimension::Y) != dimensions.end() ||
                        dimensions.find(Dimension::Z) != dimensions.end())
                    throw std::runtime_error("Missing support for spatial selection"); //TODO

                return selection != physical_parents[0];
            }
            return false;
        }
    };

    class ChooseInterpolate : public OptimizerRule {
    public:
        using OptimizerRule::OptimizerRule;

        bool visit(const logical::InterpolatedLightField &node) override {
            auto physical_parents = functional::flatmap<std::vector<PhysicalLightFieldReference>>(
                    node.parents().begin(), node.parents().end(),
                    [this](auto &parent) { return plan().assignments(parent); });

            if(physical_parents.empty())
                return false;

            //TODO clean this up, shouldn't just be randomly picking last parent
            auto hardcoded_parent = physical_parents[0].is<physical::GPUDecode>() || physical_parents[0].is<physical::CPUtoGPUTransfer>()
                                    ? physical_parents[0]
                                    : physical_parents[physical_parents.size() - 1];

            if(!plan().has_physical_assignment(node)) {
                if(hardcoded_parent->device() != physical::DeviceType::GPU)
                    throw std::runtime_error("Hardcoded support only for GPU interpolation"); //TODO
                //if(!node.interpolator()->has_implementation(physical::DeviceType::GPU)) {
                plan().emplace<physical::GPUInterpolate>(plan().lookup(node), hardcoded_parent, node.interpolator());
                return true;
            }
            return false;
        }
    };

    class ChooseDiscretize : public OptimizerRule {
        void teeIfNecessary(const LightField& node, PhysicalLightFieldReference physical) {
            auto children = plan().children(plan().lookup(node));
            if(children.size() > 1) {
                auto tees = physical::TeedPhysicalLightFieldAdapter::make(physical, children.size());
                for (auto index = 0u; index < children.size(); index++) {
                    if (physical->device() == physical::DeviceType::CPU)
                        plan().assign(plan().lookup(node), tees->physical(index));
                    else if (physical->device() == physical::DeviceType::GPU)
                        plan().emplace<physical::GPUOperatorAdapter>(tees->physical(index), physical);
                    else
                        throw InvalidArgumentError("No rule support for device type.", "node");
                }
            }
        }


    public:
        using OptimizerRule::OptimizerRule;

        bool visit(const logical::DiscreteLightField &node) override {
            auto physical_parents = functional::flatmap<std::vector<PhysicalLightFieldReference>>(
                    node.parents().begin(), node.parents().end(),
                    [this](auto &parent) { return plan().assignments(parent); });

            if (physical_parents.empty())
                return false;

            //TODO clean this up, shouldn't just be randomly picking last parent
            auto hardcoded_parent = physical_parents[0].is<physical::GPUDecode>() ||
                                    physical_parents[0].is<physical::CPUtoGPUTransfer>()
                                    ? physical_parents[0]
                                    : physical_parents[physical_parents.size() - 1];
            auto is_discrete = (hardcoded_parent.is<physical::GPUDecode>() &&
                                hardcoded_parent->logical().is<logical::ScannedLightField>()) ||
                               hardcoded_parent.is<physical::GPUDownsampleResolution>();

            if (!plan().has_physical_assignment(node) && is_discrete &&
                hardcoded_parent.is<physical::GPUDownsampleResolution>()) {
                auto &discrete = node.downcast<logical::DiscreteLightField>();
                hardcoded_parent.downcast<physical::GPUDownsampleResolution>().geometries().push_back(static_cast<IntervalGeometry&>(*discrete.geometry()));
                // Was CPUIdentity, bug or intentional?
                auto identity = plan().emplace<physical::GPUIdentity>(plan().lookup(node), hardcoded_parent);
                teeIfNecessary(node, identity);
                return true;
            } else if(!plan().has_physical_assignment(node) && is_discrete) {
                auto downsampled = hardcoded_parent->logical().try_downcast<logical::DiscretizedLightField>();
                auto scanned = downsampled.has_value() ? hardcoded_parent->logical()->parents()[0].downcast<logical::ScannedLightField>() : hardcoded_parent->logical().downcast<logical::ScannedLightField>();
                auto &parent_geometry = scanned.metadata().geometry();
                auto &discrete_geometry = node.geometry();

                if(scanned.metadata().streams().size() != 1)
                    return false;
                else if(!discrete_geometry.is<IntervalGeometry>())
                    return false;
                else if(!parent_geometry.is<EquirectangularGeometry>())
                    return false;
                else if((discrete_geometry.downcast<IntervalGeometry>().dimension() == Dimension::Theta &&
                         scanned.metadata().streams()[0].configuration().width % discrete_geometry.downcast<IntervalGeometry>().size().value_or(1) == 0) ||
                        (discrete_geometry.downcast<IntervalGeometry>().dimension() == Dimension::Phi &&
                         scanned.metadata().streams()[0].configuration().height % discrete_geometry.downcast<IntervalGeometry>().size().value_or(1) == 0))
                {
                    if(hardcoded_parent->device() == physical::DeviceType::GPU)
                    {
                        auto downsample = plan().emplace<physical::GPUDownsampleResolution>(plan().lookup(node), hardcoded_parent, discrete_geometry.downcast<IntervalGeometry>());
                        teeIfNecessary(node, downsample);
                        return true;
                    }
                }
                //TODO handle case where interval is equal to resolution (by applying identity)
            }
            return false;
        }
    };

    class ChooseLinearScale : public OptimizerRule {
    public:
        using OptimizerRule::OptimizerRule;

        bool visit(const logical::DiscreteLightField &node) override {
            auto physical_parents = functional::flatmap<std::vector<PhysicalLightFieldReference>>(
                    node.parents().begin(), node.parents().end(),
                    [this](auto &parent) { return plan().assignments(parent); });

            if(physical_parents.empty())
                return false;

            //TODO clean this up, shouldn't just be randomly picking last parent
            auto hardcoded_parent = physical_parents[0].is<physical::GPUDecode>() || physical_parents[0].is<physical::CPUtoGPUTransfer>()
                                    ? physical_parents[0]
                                    : physical_parents[physical_parents.size() - 1];
            auto hardcoded_grandparent = hardcoded_parent->parents()[0];
            auto hardcoded_greatgrandparent = hardcoded_grandparent->parents()[0];
            auto is_linear_interpolated =
                    hardcoded_parent.is<physical::GPUInterpolate>() &&
                    hardcoded_parent->logical().downcast<logical::InterpolatedLightField>().interpolator()->name() == "linear";

            if(!plan().has_physical_assignment(node) && is_linear_interpolated) {
                auto scanned = hardcoded_grandparent->logical().is<logical::ScannedLightField>() ? hardcoded_grandparent->logical().downcast<logical::ScannedLightField>() : hardcoded_greatgrandparent->logical().downcast<logical::ScannedLightField>();
                auto &scanned_geometry = scanned.metadata().geometry();
                auto &discrete_geometry = node.geometry();

                if(scanned.metadata().streams().size() != 1)
                    return false;
                else if(!discrete_geometry.is<IntervalGeometry>())
                    return false;
                else if(!scanned_geometry.is<EquirectangularGeometry>())
                    return false;
                else if(discrete_geometry.downcast<IntervalGeometry>().dimension() == Dimension::Theta ||
                        discrete_geometry.downcast<IntervalGeometry>().dimension() == Dimension::Phi)
                {
                    if(hardcoded_parent->device() == physical::DeviceType::GPU)
                    {
                        plan().emplace<physical::GPUDownsampleResolution>(plan().lookup(node), hardcoded_parent, discrete_geometry.downcast<IntervalGeometry>());
                        return true;
                    }
                }
                //TODO handle case where interval is equal to resolution (by applying identity)
            }
            return false;
        }
    };

    class ChoosePartition : public OptimizerRule {
    public:
        using OptimizerRule::OptimizerRule;

        bool visit(const logical::PartitionedLightField &node) override {
            auto physical_parents = functional::flatmap<std::vector<PhysicalLightFieldReference>>(
                    node.parents().begin(), node.parents().end(),
                    [this](auto &parent) { return plan().assignments(parent); });

            if(physical_parents.empty())
                return false;

            //TODO clean this up, shouldn't just be randomly picking last parent
            auto hardcoded_parent = physical_parents[physical_parents.size() - 1];

            if(!plan().has_physical_assignment(node)) {
                if(hardcoded_parent->device() == physical::DeviceType::CPU)
                    plan().emplace<physical::CPUIdentity>(plan().lookup(node), hardcoded_parent);
                else if(hardcoded_parent.is<physical::GPUOperator>())
                    plan().emplace<physical::GPUIdentity>(plan().lookup(node), hardcoded_parent);
                return true;
            }
            return false;
        }
    };

    class ChooseSubquery : public OptimizerRule {
        void teeIfNecessary(const LightField& node, const PhysicalLightFieldReference &physical) {
            auto children = plan().children(plan().lookup(node));
            if(children.size() > 1) {
                auto tees = physical::TeedPhysicalLightFieldAdapter::make(physical, children.size());
                for (auto index = 0u; index < children.size(); index++) {
                    plan().assign(plan().lookup(node), tees->physical(index));
                    /*if (physical->device() == physical::DeviceType::CPU)
                        plan().assign(plan().lookup(node), tees->physical(index));
                    else if (physical->device() == physical::DeviceType::GPU)
                        plan().emplace<physical::GPUOperatorAdapter>(tees->physical(index), physical);
                    else
                        throw InvalidArgumentError("No rule support for device type.", "node");*/
                }
            }
        }

    public:
        bool visit(const logical::SubqueriedLightField &node) override {
            auto physical_parents = functional::flatmap<std::vector<PhysicalLightFieldReference>>(
                    node.parents().begin(), node.parents().end(),
                    [this](auto &parent) { return plan().assignments(parent); });

            if(physical_parents.empty())
                return false;

            //TODO clean this up, shouldn't just be randomly picking last parent
            auto hardcoded_parent = physical_parents[0];

            if(!plan().has_physical_assignment(node)) {
                auto subquery = plan().emplace<physical::GPUAngularSubquery>(plan().lookup(node), hardcoded_parent,
                                                             plan().environment());
                teeIfNecessary(node, subquery);
                return true;
            }
            return false;
        }
    };

    class ChooseStore : public OptimizerRule {
    public:
        using OptimizerRule::OptimizerRule;

        PhysicalLightFieldReference Encode(const logical::StoredLightField &node, PhysicalLightFieldReference parent) {
            auto logical = plan().lookup(node);

            // Can we leverage the ChooseEncode rule to automatically do this stuff, which is an exact duplicate?

            if(parent.is<physical::GPUAngularSubquery>() && parent.downcast<physical::GPUAngularSubquery>().subqueryType().is<logical::EncodedLightField>()) {
                return plan().emplace<physical::CPUIdentity>(logical, parent);
            //} else if(parent.is<physical::GPUOperatorAdapter>() && parent->parents()[0].is<physical::GPUAngularSubquery>() && parent->parents()[0].downcast<physical::GPUAngularSubquery>().subqueryType().is<logical::EncodedLightField>()) {
            //    return plan().emplace<physical::CPUIdentity>(logical, parent);
            } else if(parent.is<physical::GPUOperator>()) {
                return plan().emplace<physical::GPUEncode>(logical, parent, Codec::hevc());
            } else if(parent.is<physical::CPUMap>() && parent.downcast<physical::CPUMap>().transform()(physical::DeviceType::CPU).codec().name() == node.codec().name()) {
                return plan().emplace<physical::CPUIdentity>(logical, parent);
                //TODO this is silly -- every physical operator should declare an output type and we should just use that
            } else if(parent.is<physical::TeedPhysicalLightFieldAdapter::TeedPhysicalLightField>() && parent->parents()[0].is<physical::CPUMap>() && parent->parents()[0].downcast<physical::CPUMap>().transform()(physical::DeviceType::CPU).codec().name() == node.codec().name()) {
                return plan().emplace<physical::CPUIdentity>(logical, parent);
            } else if(parent.is<physical::TeedPhysicalLightFieldAdapter::TeedPhysicalLightField>() && parent->parents()[0].is<physical::GPUAngularSubquery>()) {
                return plan().emplace<physical::CPUIdentity>(logical, parent);
            } else if(parent->device() != physical::DeviceType::GPU) {
                auto gpu = plan().environment().gpus()[0];
                auto transfer = plan().emplace<physical::CPUtoGPUTransfer>(logical, parent, gpu);
                return plan().emplace<physical::GPUEncode>(logical, transfer, Codec::hevc());
            } else if(!parent.is<physical::GPUOperator>()) {
                auto gpuop = plan().emplace<physical::GPUOperatorAdapter>(parent);
                return plan().emplace<physical::GPUEncode>(logical, gpuop, Codec::hevc());
            } else
                return plan().emplace<physical::GPUEncode>(logical, parent, Codec::hevc());
        }

        bool visit(const logical::StoredLightField &node) override {
            if(!plan().has_physical_assignment(node)) {
                auto physical_parents = functional::flatmap<std::vector<PhysicalLightFieldReference>>(
                        node.parents().begin(), node.parents().end(),
                        [this](auto &parent) { return plan().unassigned(parent); });

                if(physical_parents.empty())
                    return false;

                LOG(WARNING) << "Randomly picking HEVC as codec";

                auto encode = Encode(node, physical_parents[0]);
                plan().emplace<physical::Store>(plan().lookup(node), encode);
                return true;
            }
            return false;
        }
    };

    class ChooseSink : public OptimizerRule {
    public:
        using OptimizerRule::OptimizerRule;

        PhysicalLightFieldReference Encode(const logical::SunkLightField &node, PhysicalLightFieldReference parent) {
            auto logical = plan().lookup(node);

            // Can we leverage the ChooseEncode rule to automatically do this stuff, which is an exact duplicate?
            //TODO Copied from ChooseStore, which is lame

            if(parent.is<physical::GPUAngularSubquery>() && parent.downcast<physical::GPUAngularSubquery>().subqueryType().is<logical::EncodedLightField>()) {
                return plan().emplace<physical::CPUIdentity>(logical, parent);
                //} else if(parent.is<physical::GPUOperatorAdapter>() && parent->parents()[0].is<physical::GPUAngularSubquery>() && parent->parents()[0].downcast<physical::GPUAngularSubquery>().subqueryType().is<logical::EncodedLightField>()) {
                //    return plan().emplace<physical::CPUIdentity>(logical, parent);
            } else if(parent.is<physical::GPUOperator>()) {
                return plan().emplace<physical::GPUEncode>(logical, parent, Codec::hevc());
                //TODO this is silly -- every physical operator should declare an output type and we should just use that
            } else if(parent.is<physical::TeedPhysicalLightFieldAdapter::TeedPhysicalLightField>() && parent->parents()[0].is<physical::GPUAngularSubquery>()) {
                return plan().emplace<physical::CPUIdentity>(logical, parent);
            } else if(parent->device() != physical::DeviceType::GPU) {
                auto gpu = plan().environment().gpus()[0];
                auto transfer = plan().emplace<physical::CPUtoGPUTransfer>(logical, parent, gpu);
                return plan().emplace<physical::GPUEncode>(logical, transfer, Codec::hevc());
            } else if(!parent.is<physical::GPUOperator>()) {
                auto gpuop = plan().emplace<physical::GPUOperatorAdapter>(parent);
                return plan().emplace<physical::GPUEncode>(logical, gpuop, Codec::hevc());
            } else
                return plan().emplace<physical::GPUEncode>(logical, parent, Codec::hevc());
        }

        bool visit(const logical::SunkLightField &node) override {
            if(!plan().has_physical_assignment(node)) {
                auto physical_parents = functional::flatmap<std::vector<PhysicalLightFieldReference>>(
                        node.parents().begin(), node.parents().end(),
                        [this](auto &parent) { return plan().unassigned(parent); });

                if(physical_parents.empty())
                    return false;

                auto sink = Encode(node, physical_parents[0]);
                plan().emplace<physical::Sink>(plan().lookup(node), sink);
                return true;
            }
            return false;
        }
    };
}

#endif //LIGHTDB_RULES_H

#ifndef LIGHTDB_PLAN_H
#define LIGHTDB_PLAN_H

#include "LightField.h"
#include "PhysicalOperators.h"
#include "Environment.h"
#include <set>

namespace lightdb::optimization {
    class Plan {
    public:
        explicit Plan(const execution::Environment &environment, const LightFieldReference &sink)
                : Plan(environment, std::vector{sink}) {}

        explicit Plan(const execution::Environment &environment, const std::vector<LightFieldReference> &sinks)
                : Plan(environment, sinks.begin(), sinks.end()) { }

        template<typename InputIterator>
        explicit Plan(execution::Environment environment, const InputIterator first, const InputIterator last)
                : environment_(std::move(environment)),
                  sinks_(first, last),
                  physical_() {
            std::for_each(first, last, std::bind(&Plan::associate, this, std::placeholders::_1));
        }

        void associate(const LightFieldReference &node) {
            nodes_.insert(std::make_pair(node.operator->(), node));
            std::for_each(node->parents().begin(), node->parents().end(),
                          std::bind(&Plan::associate, this, std::placeholders::_1));
        }

        const PhysicalLightFieldReference& add(const PhysicalLightFieldReference &physical) {
            physical_.push_back(physical);
            assign(physical->logical(), physical);
            return physical;
        }

        template<typename PhysicalLightField, typename... Args>
        const PhysicalLightFieldReference& emplace(Args&&... args) {
            const auto &value = physical_.emplace_back(PhysicalLightFieldReference::make<PhysicalLightField>(args...));
            assign(value->logical(), value);
            return value;
        }

        void assign(const LightFieldReference &node, const PhysicalLightFieldReference &physical) {
            if(assigned_.find(&*node) == assigned_.end())
                assigned_[&*node] = {};
            assigned_[&*node].push_back(physical);
        }

        inline bool has_physical_assignment(const LightField &node) { return has_physical_assignment(lookup(node)); }

        bool has_physical_assignment(const LightFieldReference &reference) {
            return assigned_.find(&*reference) != assigned_.end();
        }

        inline auto assignments(const LightField &node) const { return assignments(lookup(node)); }

        const std::vector<PhysicalLightFieldReference> assignments(const LightFieldReference &reference) const {
            auto element = assigned_.find(&*reference);
            return element != assigned_.end()
                   ? element->second
                   : std::vector<PhysicalLightFieldReference>{};
        }

        std::vector<PhysicalLightFieldReference> assignments(const LightFieldReference &reference) {
            auto element = assigned_.find(&*reference);
            return element != assigned_.end()
                   ? element->second
                   : std::vector<PhysicalLightFieldReference>{};
        }

        std::vector<PhysicalLightFieldReference> unassigned(const LightFieldReference &reference, const bool global=true) const {
            auto assignments = this->assignments(reference);
            std::set<PhysicalLightField*> nonleafs;
            std::vector<PhysicalLightFieldReference> leafs;

            for(const auto &assignment: assignments)
                for(auto& parent: assignment->parents())
                    nonleafs.insert(&*parent);

            for(const auto &assignment: assignments)
                if(nonleafs.find(&*assignment) == nonleafs.end())
                    leafs.push_back(assignment);

            //TODO this is like O(n^999) :-/
            if(global)
                for (const auto &child: children(reference))
                    for (const auto &assignment: this->assignments(child))
                        for (const auto &parent: assignment->parents())
                            for (auto index = 0u; index < leafs.size(); index++)
                                if (parent == leafs[index])
                                    leafs.erase(leafs.begin() + index--);
            return leafs;
        }

        std::vector<LightFieldReference> children(const LightFieldReference &reference) const {
            std::vector<LightFieldReference> children;

            for(const auto &node: nodes_)
                for(const LightFieldReference &parent: node.second->parents())
                    if(parent == reference)
                        children.push_back(node.second);

            return children;
        }

        const auto& physical() const { return physical_; }
        const auto& environment() const { return environment_; }
        LightFieldReference lookup(const LightField &node) const { return nodes_.at(&node); }
        const std::vector<LightFieldReference>& sinks() const { return sinks_; }

    private:
        const execution::Environment environment_;
        std::vector<LightFieldReference> sinks_;
        //TODO can be replaced with an addressable shared_reference
        std::unordered_map<const LightField *, LightFieldReference> nodes_;
        std::unordered_map<const LightField*, std::vector<PhysicalLightFieldReference>> assigned_;
        //TODO this should be a set once rule/ordering is cleaned up
        //std::set<PhysicalLightFieldReference,
        //         bool (*)(const PhysicalLightFieldReference &, const PhysicalLightFieldReference &)> physical_;
        std::vector<PhysicalLightFieldReference> physical_;

    };
}

#endif //LIGHTDB_PLAN_H

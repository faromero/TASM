#ifndef TASM_RECTANGLE_H
#define TASM_RECTANGLE_H

#include <boost/functional/hash.hpp>

namespace tasm {
struct Rectangle {
    unsigned int id;
    unsigned int x, y;
    unsigned int width, height;

    Rectangle(unsigned int id = 0, unsigned int x = 0, unsigned int y = 0, unsigned int width = 0, unsigned int height = 0)
            : id(id), x(x % 2 ? x - 1 : x), y(y % 2 ? y - 1 : y), width(width % 2 ? width + 1 : width), height(height % 2 ? height + 1 : height)
    {}

    bool operator==(const Rectangle &other) const {
        return id == other.id
               && x == other.x
               && y == other.y
               && width == other.width
               && height == other.height;
    }

    bool hasEqualDimensions(const Rectangle &other) const {
        return width == other.width
               && height == other.height;
    }

    unsigned int area() const {
        return width * height;
    }

    bool containsPoint(const unsigned int &posX, const unsigned int &posY) const {
        return x <= posX
               && y <= posY
               && x + width > posX
               && y + height > posY;
    }

    bool intersects(const Rectangle &other) const {
        bool val = !(x >= other.x + other.width
                     || other.x >= x + width
                     || y >= other.y + other.height
                     || other.y >= y + height);

        return val;
    }

    Rectangle overlappingRectangle(const Rectangle &other) const {
        auto top = std::max(y, other.y);
        auto bottom = std::min(y + height, other.y + other.height);
        auto left = std::max(x, other.x);
        auto right = std::min(x + width, other.x + other.width);

        return Rectangle(other.id, left, top, right - left, bottom - top);
    }

    Rectangle expand(const Rectangle &other) {
        auto left = std::min(x, other.x);
        auto right = std::max(x + width, other.x + other.width);
        x = left;
        width = right - left;

        auto top = std::min(y, other.y);
        auto bottom = std::max(y + height, other.y + other.height);
        y = top;
        height = bottom - top;

        return *this;
    }
};

class RectangleMerger {
public:
    RectangleMerger(std::unique_ptr<std::list<Rectangle>> rectangles)
            : rectangles_(std::move(rectangles))
    {
        merge();
    }

    void addRectangle(const Rectangle &other) {
        auto merged = false;
        for (auto &rectangle : *rectangles_) {
            if (rectangle.intersects(other)) {
                rectangle.expand(other);
                merged = true;
            }
        }
        if (!merged)
            rectangles_->push_back(other);

        merge();
    }

    const std::list<Rectangle> &rectangles() const { return *rectangles_; }

private:
    void merge() {
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto it = rectangles_->begin(); it != rectangles_->end(); ++it) {
                for (auto it2 = std::next(it, 1); it2 != rectangles_->end(); ++it2) {
                    if (it->intersects(*it2)) {
                        it->expand(*it2);
                        rectangles_->erase(it2);
                        changed = true;
                        break;
                    }
                }
                if (changed)
                    break;
            }
        }
    }

    std::unique_ptr<std::list<Rectangle>> rectangles_;
};
} // namespace tasm

namespace std {
template<>
struct hash<tasm::Rectangle> {
    size_t operator()(const tasm::Rectangle &rect) const {
        size_t seed = 0;
        boost::hash_combine(seed, rect.id);
        boost::hash_combine(seed, rect.x);
        boost::hash_combine(seed, rect.y);
        boost::hash_combine(seed, rect.width);
        boost::hash_combine(seed, rect.height);

        return seed;
    }
};
} // namespace std

#endif //TASM_RECTANGLE_H

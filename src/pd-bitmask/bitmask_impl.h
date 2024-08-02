/// Implementation of the Bitmap Array class
/// Nothing in this mmodule may use the Python runtime
#pragma once

#include <nanoarrow/nanoarrow.hpp>

#include <memory>
#include <stdexcept>

class BitmaskArrayImpl {
public:
  // TODO: this should be private
  nanoarrow::UniqueBitmap bitmap_;

  BitmaskArrayImpl();
  explicit BitmaskArrayImpl(nanoarrow::UniqueBitmap &&bitmap);
  auto Length() const noexcept -> ssize_t;
  auto GetItem(ssize_t index) const -> bool;
  auto SetItem(ssize_t index, bool value) -> void;
  auto Invert() const noexcept -> BitmaskArrayImpl;

  template <typename OP>
  auto BinaryOp(const BitmaskArrayImpl &other, OP op) const
      -> BitmaskArrayImpl {
    if (bitmap_->size_bits != other.bitmap_->size_bits) {
      throw std::invalid_argument(
          "Shape of other does not match bitmask shape");
    }

    nanoarrow::UniqueBitmap new_bitmap;
    const size_t nbits = bitmap_->size_bits;

    ArrowBitmapInit(new_bitmap.get());
    ArrowBitmapReserve(new_bitmap.get(), nbits);

    const size_t size_bytes = bitmap_->buffer.size_bytes;
    const size_t overflow_limit = SIZE_MAX - sizeof(size_t);
    const size_t limit =
        size_bytes > overflow_limit ? overflow_limit : size_bytes;

    int64_t i = 0;
    for (; i + sizeof(int64_t) - 1 < limit; i += sizeof(int64_t)) {
      uint64_t value1;
      uint64_t value2;
      uint64_t result;
      memcpy(&value1, &bitmap_->buffer.data[i], sizeof(uint64_t));
      memcpy(&value2, &other.bitmap_->buffer.data[i], sizeof(uint64_t));
      result = op(value1, value2);
      memcpy(&new_bitmap->buffer.data[i], &result, sizeof(uint64_t));
    }

    for (; i < bitmap_->buffer.size_bytes; i++) {
      new_bitmap->buffer.data[i] =
          op(bitmap_->buffer.data[i], other.bitmap_->buffer.data[i]);
    }

    new_bitmap->size_bits = bitmap_->size_bits;
    new_bitmap->buffer.size_bytes = bitmap_->buffer.size_bytes;

    return BitmaskArrayImpl(std::move(new_bitmap));
  }

  auto Size() const noexcept -> ssize_t;
  auto NBytes() const noexcept -> ssize_t;
  auto Any() const noexcept -> bool;
  auto All() const noexcept -> bool;
  auto Sum() const noexcept -> ssize_t;

  auto Copy() const noexcept -> BitmaskArrayImpl;

  class iterator {
  public:
    explicit iterator(const BitmaskArrayImpl &bmai, int curr_index = 0)
        : bmai_(bmai), curr_index_(curr_index) {}

    iterator &operator++() {
      ++curr_index_;
      return *this;
    }

    bool operator==(iterator other) const {
      return &bmai_ == &other.bmai_ && curr_index_ == other.curr_index_;
    }

    bool operator!=(iterator other) const { return !(*this == other); }

    bool operator*() { return bmai_.GetItem(curr_index_); }

  private:
    const BitmaskArrayImpl &bmai_;
    int curr_index_ = 0;
  };

  iterator begin() const noexcept { return iterator(*this); }
  iterator end() const noexcept { return iterator(*this, Length()); }
};

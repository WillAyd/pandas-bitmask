/// Implementation of the Bitmap Array class
/// Nothing in this mmodule may use the Python runtime
#include "bitmask_impl.h"
#include "buffer_inline.h"

BitmaskArrayImpl::BitmaskArrayImpl() = default;
BitmaskArrayImpl::BitmaskArrayImpl(nanoarrow::UniqueBitmap &&bitmap)
    : bitmap_(std::move(bitmap)) {}
auto BitmaskArrayImpl::Length() const noexcept -> ssize_t {
  return bitmap_->size_bits;
}
auto BitmaskArrayImpl::GetItem(ssize_t index) const -> bool {
  if (index < 0) {
    index += bitmap_->size_bits;
    if (index < 0) {
      throw std::out_of_range("index out of range");
    }
  }
  if (index >= bitmap_->size_bits) {
    throw std::out_of_range("index out of range");
  }

  return ArrowBitGet(bitmap_->buffer.data, index);
}

auto BitmaskArrayImpl::SetItem(ssize_t index, bool value) -> void {
  if (index < 0) {
    index += bitmap_->size_bits;
    if (index < 0) {
      throw std::out_of_range("index out of range");
    }
  }
  if (index >= bitmap_->size_bits) {
    throw std::out_of_range("index out of range");
  }

  ArrowBitSetTo(bitmap_->buffer.data, index, value);
}

auto BitmaskArrayImpl::Invert() const noexcept -> BitmaskArrayImpl {
  nanoarrow::UniqueBitmap new_bitmap;
  const size_t nbits = bitmap_->size_bits;

  ArrowBitmapInit(new_bitmap.get());
  ArrowBitmapReserve(new_bitmap.get(), nbits);

  const int64_t size_bytes = bitmap_->buffer.size_bytes;
  const int64_t overflow_limit = INT64_MAX - sizeof(int64_t);
  const int64_t limit =
      size_bytes > overflow_limit ? overflow_limit : size_bytes;

  int64_t i = 0;
  for (; i + static_cast<int64_t>(sizeof(int64_t)) - 1 < limit;
       i += sizeof(int64_t)) {
    uint64_t value;
    memcpy(&value, &bitmap_->buffer.data[i], sizeof(uint64_t));
    value = ~value;
    memcpy(&new_bitmap->buffer.data[i], &value, sizeof(uint64_t));
  }

  for (; i < bitmap_->buffer.size_bytes; i++) {
    new_bitmap->buffer.data[i] = ~bitmap_->buffer.data[i];
  }

  new_bitmap->size_bits = nbits;
  return BitmaskArrayImpl(std::move(new_bitmap));
}

auto BitmaskArrayImpl::Size() const noexcept -> ssize_t {
  return bitmap_->size_bits;
}

auto BitmaskArrayImpl::NBytes() const noexcept -> ssize_t {
  return bitmap_->buffer.size_bytes;
}

auto BitmaskArrayImpl::Any() const noexcept -> bool {
  const int64_t nbits = bitmap_->size_bits;
  if (nbits < 1) {
    return false;
  }

  const int64_t size_bytes = bitmap_->buffer.size_bytes;
  const int64_t overflow_limit = INT64_MAX - sizeof(int64_t);
  const int64_t limit =
      size_bytes > overflow_limit ? overflow_limit : size_bytes;
  int64_t i = 0;
  for (; i + static_cast<int64_t>(sizeof(int64_t)) - 1 < limit;
       i += sizeof(int64_t)) {
    uint64_t value;
    memcpy(&value, &bitmap_->buffer.data[i], sizeof(uint64_t));
    if (value != 0x0) {
      return true;
    }
  }

  for (; i < bitmap_->buffer.size_bytes - 1; i++) {
    if (bitmap_->buffer.data[i] != 0x0) {
      return true;
    }
  }

  const int64_t bits_remaining = nbits - ((size_bytes - 1) * 8);
  for (int64_t i = 0; i < bits_remaining; i++) {
    if (ArrowBitGet(bitmap_->buffer.data, nbits - i - 1) == 1) {
      return true;
    }
  }

  return false;
}

auto BitmaskArrayImpl::All() const noexcept -> bool {
  const int64_t nbits = bitmap_->size_bits;
  if (nbits < 1) {
    return true;
  }

  const int64_t size_bytes = bitmap_->buffer.size_bytes;
  const int64_t overflow_limit = INT64_MAX - sizeof(int64_t);
  const int64_t limit =
      size_bytes > overflow_limit ? overflow_limit : size_bytes;
  int64_t i = 0;
  for (; i + static_cast<int64_t>(sizeof(int64_t)) - 1 < limit;
       i += sizeof(int64_t)) {
    uint64_t value;
    memcpy(&value, &bitmap_->buffer.data[i], sizeof(uint64_t));
    if (value != UINT64_MAX) {
      return false;
    }
  }

  for (; i < bitmap_->buffer.size_bytes - 1; i++) {
    if (bitmap_->buffer.data[i] != 0xff) {
      return false;
    }
  }

  const size_t bits_remaining = nbits - ((size_bytes - 1) * 8);
  for (size_t i = 0; i < bits_remaining; i++) {
    if (ArrowBitGet(bitmap_->buffer.data, nbits - i - 1) == 0) {
      return false;
    }
  }

  return true;
}

auto BitmaskArrayImpl::Sum() const noexcept -> ssize_t {
  return static_cast<ssize_t>(
      ArrowBitCountSet(bitmap_->buffer.data, 0, bitmap_->size_bits));
}

auto BitmaskArrayImpl::Copy() const noexcept -> BitmaskArrayImpl {
  nanoarrow::UniqueBitmap new_bitmap;
  const size_t nbits = bitmap_->size_bits;

  ArrowBitmapInit(new_bitmap.get());
  ArrowBitmapReserve(new_bitmap.get(), nbits);
  memcpy(new_bitmap->buffer.data, bitmap_->buffer.data,
         bitmap_->buffer.size_bytes);

  new_bitmap->size_bits = nbits;
  return BitmaskArrayImpl(std::move(new_bitmap));
}

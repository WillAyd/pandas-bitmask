/// Implementation of the Bitmap Array class
/// Nothing in this mmodule may use the Python runtime
#include "bitmask_impl.h"

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

  const size_t n_uint64s = nbits / sizeof(uint64_t);
  const size_t rem = (nbits % sizeof(int64_t) + 7) / 8;

  const size_t size_bytes = bitmap_->buffer.size_bytes;
  const size_t overflow_limit = SIZE_MAX - sizeof(size_t);
  const size_t limit =
      size_bytes > overflow_limit ? overflow_limit : size_bytes;

  int64_t i = 0;
  for (; i + sizeof(int64_t) - 1 < limit; i += sizeof(int64_t)) {
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

auto BitmaskArrayImpl::ExposeBufferForPython() noexcept -> std::byte * {
  const auto nelems = this->Length();
  py_buffer_ = new std::byte[nelems];
  ArrowBitsUnpackInt8(bitmap_->buffer.data, 0, nelems,
                      reinterpret_cast<int8_t *>(py_buffer_));

  return py_buffer_;
}

auto BitmaskArrayImpl::ReleasePyBuffer() noexcept -> void {
  delete[] py_buffer_;
}

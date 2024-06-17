/// Implementation of the Bitmap Array class
/// Nothing in this mmodule may use the Python runtime
#include "bitmask_impl.h"

#include <stdexcept>

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

auto BitmaskArrayImpl::Invert() const noexcept -> BitmaskArrayImpl {
  nanoarrow::UniqueBitmap new_bitmap;
  const size_t nbits = bitmap_->size_bits;

  ArrowBitmapInit(new_bitmap.get());
  ArrowBitmapReserve(new_bitmap.get(), nbits);

  const size_t n_uint64s = nbits / sizeof(uint64_t);
  const size_t rem = (nbits % sizeof(int64_t) + 7) / 8;
  uint8_t *src = bitmap_->buffer.data;
  uint8_t *dst = new_bitmap->buffer.data;

  for (size_t i = 0; i < n_uint64s; ++i) {
    uint64_t value;
    memcpy(&value, src, sizeof(uint64_t));
    value = ~value;
    memcpy(dst, &value, sizeof(uint64_t));
    src += sizeof(uint64_t);
    dst += sizeof(uint64_t);
  }

  for (size_t i = 0; i < rem; ++i) {
    uint8_t value;
    memcpy(&value, src, sizeof(uint8_t));
    value = ~value;
    memcpy(dst, &value, sizeof(uint8_t));
    src += sizeof(uint8_t);
    dst += sizeof(uint8_t);
  }

  new_bitmap->size_bits = nbits;
  return BitmaskArrayImpl(std::move(new_bitmap));
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

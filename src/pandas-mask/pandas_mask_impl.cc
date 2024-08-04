/// Implementation of the Bitmap Array class
/// Nothing in this mmodule may use the Python runtime
#include "pandas_mask_impl.h"
#include "nanoarrow.h"

PandasMaskArrayImpl::PandasMaskArrayImpl() = default;
PandasMaskArrayImpl::PandasMaskArrayImpl(nanoarrow::UniqueBitmap &&bitmap)
    : bitmap_(std::move(bitmap)) {}
auto PandasMaskArrayImpl::Length() const noexcept -> ssize_t {
  return bitmap_->size_bits;
}
auto PandasMaskArrayImpl::GetItem(ssize_t index) const -> bool {
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

auto PandasMaskArrayImpl::GetItem(std::vector<ssize_t> values) const
    -> PandasMaskArrayImpl {
  nanoarrow::UniqueBitmap new_bitmap;
  ArrowBitmapInit(new_bitmap.get());
  ArrowBitmapReserve(new_bitmap.get(), values.size());

  for (const auto idx : values) {
    const auto bit = GetItem(idx);
    ArrowBitmapAppendUnsafe(new_bitmap.get(), bit, 1);
  }

  return PandasMaskArrayImpl(std::move(new_bitmap));
}

auto PandasMaskArrayImpl::SetItem(ssize_t index, bool value) -> void {
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

auto PandasMaskArrayImpl::Invert() const noexcept -> PandasMaskArrayImpl {
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

  new_bitmap->buffer.size_bytes = bitmap_->buffer.size_bytes;
  new_bitmap->size_bits = nbits;
  return PandasMaskArrayImpl(std::move(new_bitmap));
}

auto PandasMaskArrayImpl::Size() const noexcept -> ssize_t {
  return bitmap_->size_bits;
}

auto PandasMaskArrayImpl::NBytes() const noexcept -> ssize_t {
  return bitmap_->buffer.size_bytes;
}

auto PandasMaskArrayImpl::Any() const noexcept -> bool {
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

auto PandasMaskArrayImpl::All() const noexcept -> bool {
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

auto PandasMaskArrayImpl::Sum() const noexcept -> ssize_t {
  return static_cast<ssize_t>(
      ArrowBitCountSet(bitmap_->buffer.data, 0, bitmap_->size_bits));
}

auto PandasMaskArrayImpl::Copy() const noexcept -> PandasMaskArrayImpl {
  nanoarrow::UniqueBitmap new_bitmap;
  const size_t nbits = bitmap_->size_bits;

  ArrowBitmapInit(new_bitmap.get());
  ArrowBitmapReserve(new_bitmap.get(), nbits);
  memcpy(new_bitmap->buffer.data, bitmap_->buffer.data,
         bitmap_->buffer.size_bytes);

  new_bitmap->buffer.size_bytes = bitmap_->buffer.size_bytes;
  new_bitmap->size_bits = nbits;
  return PandasMaskArrayImpl(std::move(new_bitmap));
}

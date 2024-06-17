/// Implementation of the Bitmap Array class
/// Nothing in this mmodule may use the Python runtime
#pragma once

#include <nanoarrow/nanoarrow.hpp>

#include <memory>

class BitmaskArrayImpl {
public:
  // TODO: this should be private
  nanoarrow::UniqueBitmap bitmap_;

  BitmaskArrayImpl();
  explicit BitmaskArrayImpl(nanoarrow::UniqueBitmap &&bitmap);
  auto Length() const noexcept -> ssize_t;
  auto GetItem(ssize_t index) const -> bool;
  auto Invert() const noexcept -> BitmaskArrayImpl;

  auto ExposeBufferForPython() noexcept -> std::byte *;
  auto ReleasePyBuffer() noexcept -> void;

private:
  std::byte *py_buffer_; // Python buffer protocol does not support bits
};

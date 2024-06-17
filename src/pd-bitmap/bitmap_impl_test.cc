#include "bitmap_impl.h"

#include <gtest/gtest.h>
#include <nanoarrow/nanoarrow.hpp>

TEST(BitmaskArrayImplTest, BitmapConstructor) {
  nanoarrow::UniqueBitmap bitmap;
  ArrowBitmapInit(bitmap.get());

  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 2));

  const auto bma = BitmaskArrayImpl(std::move(bitmap));
  ASSERT_EQ(bma.GetItem(0), true);
  ASSERT_EQ(bma.GetItem(1), false);
  ASSERT_EQ(bma.GetItem(2), true);
  ASSERT_EQ(bma.GetItem(3), true);
}

TEST(BitmaskArrayImplTest, Length) {
  nanoarrow::UniqueBitmap bitmap;
  ArrowBitmapInit(bitmap.get());

  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 2));

  const auto bma = BitmaskArrayImpl(std::move(bitmap));
  ASSERT_EQ(bma.Length(), 4);
}

TEST(BitmaskArrayImplTest, Invert) {
  nanoarrow::UniqueBitmap bitmap;
  ArrowBitmapInit(bitmap.get());

  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 2));

  const auto bma = BitmaskArrayImpl(std::move(bitmap));
  const auto inverted = bma.Invert();

  ASSERT_EQ(inverted.GetItem(0), false);
  ASSERT_EQ(inverted.GetItem(1), true);
  ASSERT_EQ(inverted.GetItem(2), false);
  ASSERT_EQ(inverted.GetItem(3), false);
}

TEST(BitmaskArrayImplTest, ExposeBufferForPython) {
  nanoarrow::UniqueBitmap bitmap;
  ArrowBitmapInit(bitmap.get());

  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 2));

  auto bma = BitmaskArrayImpl(std::move(bitmap));
  const auto py_buffer = bma.ExposeBufferForPython();

  ASSERT_EQ(static_cast<unsigned int>(py_buffer[0]), 1);
  ASSERT_EQ(static_cast<unsigned int>(py_buffer[1]), 0);
  ASSERT_EQ(static_cast<unsigned int>(py_buffer[2]), 1);
  ASSERT_EQ(static_cast<unsigned int>(py_buffer[3]), 1);

  bma.ReleasePyBuffer();
}

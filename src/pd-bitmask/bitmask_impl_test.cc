#include "bitmask_impl.h"

#include <gtest/gtest.h>

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

TEST(BitmaskArrayImplTest, BitmapSetItemBasic) {
  nanoarrow::UniqueBitmap bitmap;
  ArrowBitmapInit(bitmap.get());

  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 2));

  auto bma = BitmaskArrayImpl(std::move(bitmap));
  ASSERT_EQ(bma.GetItem(1), false);
  ASSERT_EQ(bma.GetItem(2), true);
  bma.SetItem(1, true);
  bma.SetItem(2, false);
  ASSERT_EQ(bma.GetItem(1), true);
  ASSERT_EQ(bma.GetItem(2), false);
}

TEST(BitmaskArrayImplTest, BitmapSetItemNegative) {
  nanoarrow::UniqueBitmap bitmap;
  ArrowBitmapInit(bitmap.get());

  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 2));

  auto bma = BitmaskArrayImpl(std::move(bitmap));
  ASSERT_EQ(bma.GetItem(-3), false);
  ASSERT_EQ(bma.GetItem(-2), true);
  bma.SetItem(-3, true);
  bma.SetItem(-2, false);
  ASSERT_EQ(bma.GetItem(-3), true);
  ASSERT_EQ(bma.GetItem(-2), false);
}

TEST(BitmaskArrayImplTest, BitmapSetItemErrors) {
  nanoarrow::UniqueBitmap bitmap;
  ArrowBitmapInit(bitmap.get());

  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 2));

  auto bma = BitmaskArrayImpl(std::move(bitmap));
  EXPECT_NO_THROW(bma.SetItem(3, true));
  EXPECT_THROW(bma.SetItem(4, true), std::out_of_range);
  EXPECT_NO_THROW(bma.SetItem(-4, true));
  EXPECT_THROW(bma.SetItem(-5, true), std::out_of_range);
}

TEST(BitmaskArrayImplTest, Invert) {
  nanoarrow::UniqueBitmap bitmap;
  ArrowBitmapInit(bitmap.get());

  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 2));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 5));

  const auto bma = BitmaskArrayImpl(std::move(bitmap));
  const auto inverted = bma.Invert();

  ASSERT_EQ(inverted.GetItem(0), false);
  ASSERT_EQ(inverted.GetItem(1), true);
  ASSERT_EQ(inverted.GetItem(2), false);
  ASSERT_EQ(inverted.GetItem(3), false);
  ASSERT_EQ(inverted.GetItem(8), false);
}

TEST(BitmaskArrayImplTest, And) {
  nanoarrow::UniqueBitmap bitmap;
  ArrowBitmapInit(bitmap.get());

  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 2));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 5));

  const auto bma = BitmaskArrayImpl(std::move(bitmap));

  nanoarrow::UniqueBitmap other;
  ArrowBitmapInit(other.get());

  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 2));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 5));

  const auto bma_other = BitmaskArrayImpl(std::move(bitmap));
  const auto anded = bma.And(bma_other);

  ASSERT_EQ(anded.GetItem(0), true);
  ASSERT_EQ(anded.GetItem(1), false);
  ASSERT_EQ(anded.GetItem(2), false);
  ASSERT_EQ(anded.GetItem(3), false);
  ASSERT_EQ(anded.GetItem(8), true);
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

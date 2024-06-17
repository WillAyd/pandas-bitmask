#include "bitmask_impl.h"

#include <gtest/gtest.h>

#include <functional>

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

class BitmaskArrayBinaryOpTest : public testing::Test {
protected:
  BitmaskArrayBinaryOpTest() {
    nanoarrow::UniqueBitmap bitmap;
    ArrowBitmapInit(bitmap.get());

    NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
    NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 1));
    NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 2));
    NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 5));

    bma1_ = BitmaskArrayImpl(std::move(bitmap));

    nanoarrow::UniqueBitmap other;
    ArrowBitmapInit(other.get());

    NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
    NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
    NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 2));
    NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 5));

    bma2_ = BitmaskArrayImpl(std::move(bitmap));
  }

  BitmaskArrayImpl bma1_, bma2_;
};

TEST_F(BitmaskArrayBinaryOpTest, Add) {
  const auto anded = bma1_.BinaryOp(bma2_, std::bit_and());

  ASSERT_EQ(anded.GetItem(0), true);
  ASSERT_EQ(anded.GetItem(1), false);
  ASSERT_EQ(anded.GetItem(2), false);
  ASSERT_EQ(anded.GetItem(3), false);
  ASSERT_EQ(anded.GetItem(8), true);
}

TEST_F(BitmaskArrayBinaryOpTest, Or) {
  const auto ored = bma1_.BinaryOp(bma2_, std::bit_or());

  ASSERT_EQ(ored.GetItem(0), true);
  ASSERT_EQ(ored.GetItem(1), true);
  ASSERT_EQ(ored.GetItem(2), true);
  ASSERT_EQ(ored.GetItem(3), true);
  ASSERT_EQ(ored.GetItem(8), true);
}

TEST_F(BitmaskArrayBinaryOpTest, XOr) {
  const auto xored = bma1_.BinaryOp(bma2_, std::bit_xor());

  ASSERT_EQ(xored.GetItem(0), false);
  ASSERT_EQ(xored.GetItem(1), true);
  ASSERT_EQ(xored.GetItem(2), true);
  ASSERT_EQ(xored.GetItem(3), true);
  ASSERT_EQ(xored.GetItem(8), false);
}

TEST(BitmaskArrayImplTest, Size) {
  nanoarrow::UniqueBitmap bitmap;
  ArrowBitmapInit(bitmap.get());

  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 2));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 5));

  const auto bma = BitmaskArrayImpl(std::move(bitmap));
  const auto size = bma.Size();
  ASSERT_EQ(size, 9);
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

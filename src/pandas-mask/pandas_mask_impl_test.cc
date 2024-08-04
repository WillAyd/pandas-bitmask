#include "pandas_mask_impl.h"

#include <gtest/gtest.h>

#include <functional>

TEST(PandasMaskArrayImplTest, BitmapConstructor) {
  nanoarrow::UniqueBitmap bitmap;
  ArrowBitmapInit(bitmap.get());

  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 2));

  const auto bma = PandasMaskArrayImpl(std::move(bitmap));
  ASSERT_EQ(bma.GetItem(0), true);
  ASSERT_EQ(bma.GetItem(1), false);
  ASSERT_EQ(bma.GetItem(2), true);
  ASSERT_EQ(bma.GetItem(3), true);
}

TEST(PandasMaskArrayImplTest, Length) {
  nanoarrow::UniqueBitmap bitmap;
  ArrowBitmapInit(bitmap.get());

  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 2));

  const auto bma = PandasMaskArrayImpl(std::move(bitmap));
  ASSERT_EQ(bma.Length(), 4);
}

TEST(PandasMaskArrayImplTest, BitmapSetItemBasic) {
  nanoarrow::UniqueBitmap bitmap;
  ArrowBitmapInit(bitmap.get());

  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 2));

  auto bma = PandasMaskArrayImpl(std::move(bitmap));
  ASSERT_EQ(bma.GetItem(1), false);
  ASSERT_EQ(bma.GetItem(2), true);
  bma.SetItem(1, true);
  bma.SetItem(2, false);
  ASSERT_EQ(bma.GetItem(1), true);
  ASSERT_EQ(bma.GetItem(2), false);
}

TEST(PandasMaskArrayImplTest, BitmapSetItemNegative) {
  nanoarrow::UniqueBitmap bitmap;
  ArrowBitmapInit(bitmap.get());

  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 2));

  auto bma = PandasMaskArrayImpl(std::move(bitmap));
  ASSERT_EQ(bma.GetItem(-3), false);
  ASSERT_EQ(bma.GetItem(-2), true);
  bma.SetItem(-3, true);
  bma.SetItem(-2, false);
  ASSERT_EQ(bma.GetItem(-3), true);
  ASSERT_EQ(bma.GetItem(-2), false);
}

TEST(PandasMaskArrayImplTest, BitmapSetItemErrors) {
  nanoarrow::UniqueBitmap bitmap;
  ArrowBitmapInit(bitmap.get());

  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 2));

  auto bma = PandasMaskArrayImpl(std::move(bitmap));
  EXPECT_NO_THROW(bma.SetItem(3, true));
  EXPECT_THROW(bma.SetItem(4, true), std::out_of_range);
  EXPECT_NO_THROW(bma.SetItem(-4, true));
  EXPECT_THROW(bma.SetItem(-5, true), std::out_of_range);
}

TEST(PandasMaskArrayImplTest, Invert) {
  nanoarrow::UniqueBitmap bitmap;
  ArrowBitmapInit(bitmap.get());

  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 2));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 5));

  const auto bma = PandasMaskArrayImpl(std::move(bitmap));
  const auto inverted = bma.Invert();

  ASSERT_EQ(inverted.GetItem(0), false);
  ASSERT_EQ(inverted.GetItem(1), true);
  ASSERT_EQ(inverted.GetItem(2), false);
  ASSERT_EQ(inverted.GetItem(3), false);
  ASSERT_EQ(inverted.GetItem(8), false);

  ASSERT_EQ(inverted.bitmap_->buffer.size_bytes, 2);
  ASSERT_EQ(inverted.bitmap_->size_bits, 9);
}

class PandasMaskArrayBinaryOpTest : public testing::Test {
protected:
  PandasMaskArrayBinaryOpTest() {
    nanoarrow::UniqueBitmap bitmap;
    ArrowBitmapInit(bitmap.get());

    NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
    NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 1));
    NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 2));
    NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 5));

    bma1_ = PandasMaskArrayImpl(std::move(bitmap));

    nanoarrow::UniqueBitmap other;
    ArrowBitmapInit(other.get());

    NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
    NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
    NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 2));
    NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 5));

    bma2_ = PandasMaskArrayImpl(std::move(bitmap));
  }

  PandasMaskArrayImpl bma1_, bma2_;
};

TEST_F(PandasMaskArrayBinaryOpTest, Add) {
  const auto anded = bma1_.BinaryOp(bma2_, std::bit_and());

  ASSERT_EQ(anded.GetItem(0), true);
  ASSERT_EQ(anded.GetItem(1), false);
  ASSERT_EQ(anded.GetItem(2), false);
  ASSERT_EQ(anded.GetItem(3), false);
  ASSERT_EQ(anded.GetItem(8), true);
}

TEST_F(PandasMaskArrayBinaryOpTest, Or) {
  const auto ored = bma1_.BinaryOp(bma2_, std::bit_or());

  ASSERT_EQ(ored.GetItem(0), true);
  ASSERT_EQ(ored.GetItem(1), true);
  ASSERT_EQ(ored.GetItem(2), true);
  ASSERT_EQ(ored.GetItem(3), true);
  ASSERT_EQ(ored.GetItem(8), true);
}

TEST_F(PandasMaskArrayBinaryOpTest, XOr) {
  const auto xored = bma1_.BinaryOp(bma2_, std::bit_xor());

  ASSERT_EQ(xored.GetItem(0), false);
  ASSERT_EQ(xored.GetItem(1), true);
  ASSERT_EQ(xored.GetItem(2), true);
  ASSERT_EQ(xored.GetItem(3), true);
  ASSERT_EQ(xored.GetItem(8), false);
}

TEST(PandasMaskArrayImplTest, Size) {
  nanoarrow::UniqueBitmap bitmap;
  ArrowBitmapInit(bitmap.get());

  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 2));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 5));

  const auto bma = PandasMaskArrayImpl(std::move(bitmap));
  const auto size = bma.Size();
  ASSERT_EQ(size, 9);
}

TEST(PandasMaskArrayImplTest, NBytes) {
  nanoarrow::UniqueBitmap bitmap;
  ArrowBitmapInit(bitmap.get());

  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 2));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 5));

  const auto bma = PandasMaskArrayImpl(std::move(bitmap));
  const auto size = bma.NBytes();
  ASSERT_EQ(size, 2);
}

class PandasMaskArrayAnyAllTest : public testing::Test {
protected:
  PandasMaskArrayAnyAllTest() {
    nanoarrow::UniqueBitmap first;
    ArrowBitmapInit(first.get());

    NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(first.get(), 1, 7));
    NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(first.get(), 0, 8));
    bma1_ = PandasMaskArrayImpl(std::move(first));

    nanoarrow::UniqueBitmap second;
    ArrowBitmapInit(second.get());

    NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(second.get(), 1, 15));
    bma2_ = PandasMaskArrayImpl(std::move(second));

    nanoarrow::UniqueBitmap third;
    ArrowBitmapInit(third.get());

    NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(third.get(), 0, 15));
    bma3_ = PandasMaskArrayImpl(std::move(third));

    nanoarrow::UniqueBitmap fourth;
    ArrowBitmapInit(fourth.get());

    bma4_ = PandasMaskArrayImpl(std::move(fourth));
  }

  PandasMaskArrayImpl bma1_, bma2_, bma3_, bma4_;
};

TEST_F(PandasMaskArrayAnyAllTest, All) {
  ASSERT_FALSE(bma1_.All());
  ASSERT_TRUE(bma2_.All());
  ASSERT_FALSE(bma3_.All());
  ASSERT_TRUE(bma4_.All());
}

TEST_F(PandasMaskArrayAnyAllTest, Any) {
  ASSERT_TRUE(bma1_.Any());
  ASSERT_TRUE(bma2_.Any());
  ASSERT_FALSE(bma3_.Any());
  ASSERT_FALSE(bma4_.Any());
}

TEST(PandasMaskArrayImplTest, Sum) {
  nanoarrow::UniqueBitmap bitmap;
  ArrowBitmapInit(bitmap.get());

  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 2));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 5));

  const auto bma = PandasMaskArrayImpl(std::move(bitmap));
  ASSERT_EQ(bma.Sum(), 8);
}

TEST(PandasMaskArrayImplTest, Copy) {
  nanoarrow::UniqueBitmap bitmap;
  ArrowBitmapInit(bitmap.get());

  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 2));

  const auto bma = PandasMaskArrayImpl(std::move(bitmap));
  const auto copied = bma.Copy();

  ASSERT_EQ(copied.GetItem(0), true);
  ASSERT_EQ(copied.GetItem(1), false);
  ASSERT_EQ(copied.GetItem(2), true);
  ASSERT_EQ(copied.GetItem(3), true);

  ASSERT_EQ(copied.bitmap_->buffer.size_bytes, 1);
  ASSERT_EQ(copied.bitmap_->size_bits, 4);
}

TEST(PandasMaskArrayImplTest, Iteration) {
  nanoarrow::UniqueBitmap bitmap;
  ArrowBitmapInit(bitmap.get());

  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 0, 1));
  NANOARROW_THROW_NOT_OK(ArrowBitmapAppend(bitmap.get(), 1, 2));

  const auto bma = PandasMaskArrayImpl(std::move(bitmap));

  size_t idx = 0;
  for (auto value : bma) {
    if (idx == 1) {
      ASSERT_FALSE(value);
    } else {
      ASSERT_TRUE(value);
    }
    ++idx;
  }
}

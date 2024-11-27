#include "pandas_mask_impl.h"

#include <functional>
#include <sstream>
#include <string>

#include <nanoarrow/nanoarrow.h>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

using np_arr_type = nb::ndarray<nb::numpy, bool, nb::shape<-1>>;

class PandasMaskArray {
public:
  // We use a pImpl for anything that can be implemented without
  // the Python runtime. This makes it easier to test at a low level
  // and use tools like ASAN
  std::unique_ptr<PandasMaskArrayImpl> pImpl_;

  explicit PandasMaskArray(PandasMaskArrayImpl &&bmi)
      : pImpl_(std::make_unique<PandasMaskArrayImpl>(std::move(bmi))) {}

  explicit PandasMaskArray(np_arr_type np_array) {
    pImpl_ = std::make_unique<PandasMaskArrayImpl>(PandasMaskArrayImpl());
    const auto vw = np_array.view();
    const auto nelems = vw.shape(0);

    ArrowBitmapInit(pImpl_->bitmap_.get());
    NANOARROW_THROW_NOT_OK(ArrowBitmapReserve(pImpl_->bitmap_.get(), nelems));
    ArrowBitmapAppendInt8Unsafe(
        pImpl_->bitmap_.get(),
        reinterpret_cast<const int8_t *>(np_array.data()), nelems);
  }

  explicit PandasMaskArray(nanoarrow::UniqueBitmap &&bitmap)
      : pImpl_(std::make_unique<PandasMaskArrayImpl>(
            PandasMaskArrayImpl(std::move(bitmap)))) {}

  PandasMaskArray(const PandasMaskArray &pma)
      : pImpl_(std::make_unique<PandasMaskArrayImpl>(pma.pImpl_->Copy())) {}

  auto GetItem(nb::object indexer_obj) -> nb::object {
    // Simple case - support integer scalar
    ssize_t i;
    if (nb::try_cast(indexer_obj, i, false)) {
      return nb::bool_(pImpl_->GetItem(i));
    }

    // List of values
    std::vector<ssize_t> values;
    if (nb::try_cast(indexer_obj, values, false)) {
      auto *pma = new PandasMaskArray(pImpl_->GetItem(values));

      nb::handle py_type = nb::type<PandasMaskArray>();
      return nb::inst_take_ownership(py_type, pma);
    }

    // Boolean ndarray
    np_arr_type bools;
    if (nb::try_cast(indexer_obj, bools, false)) {
      if (static_cast<ssize_t>(bools.size()) != pImpl_->Length()) {
        throw nb::value_error(
            "Boolean array indexer must be same size as PandasMask");
      }

      const auto vw = bools.view();
      nanoarrow::UniqueBitmap new_bitmap;
      ArrowBitmapInit(new_bitmap.get());
      ArrowBitmapReserve(new_bitmap.get(), bools.size());

      for (ssize_t idx = 0; idx < static_cast<ssize_t>(vw.shape(0)); ++idx) {
        if (vw(idx)) {
          const auto is_set = pImpl_->GetItem(idx);
          ArrowBitmapAppendUnsafe(new_bitmap.get(), is_set, 1);
        }
      }

      auto *pma = new PandasMaskArray(std::move(new_bitmap));
      nb::handle py_type = nb::type<PandasMaskArray>();
      return nb::inst_take_ownership(py_type, pma);
    }

    // Indexing ndarray
    nb::ndarray<const ssize_t, nb::ndim<1>> indices;
    if (nb::try_cast(indexer_obj, indices, false)) {
      const auto vw = indices.view();
      nanoarrow::UniqueBitmap new_bitmap;
      ArrowBitmapInit(new_bitmap.get());
      ArrowBitmapReserve(new_bitmap.get(), indices.size());

      for (ssize_t idx = 0; idx < static_cast<ssize_t>(vw.shape(0)); ++idx) {
        const auto pos = vw(idx);
        const auto is_set = pImpl_->GetItem(pos);
        ArrowBitmapAppend(new_bitmap.get(), is_set, 1);
      }

      auto *pma = new PandasMaskArray(std::move(new_bitmap));
      nb::handle py_type = nb::type<PandasMaskArray>();
      return nb::inst_take_ownership(py_type, pma);
    }

    // slice
    nb::slice slice_obj;
    if (nb::try_cast(indexer_obj, slice_obj, false)) {
      const auto converted_slice = slice_obj.compute(pImpl_->Length());
      auto [start, stop, step, length] = converted_slice;

      nanoarrow::UniqueBitmap new_bitmap;
      ArrowBitmapInit(new_bitmap.get());
      ArrowBitmapReserve(new_bitmap.get(), length);

      for (size_t i = 0; i < length; ++i) {
        const auto is_set = pImpl_->GetItem(start);
        ArrowBitmapAppendUnsafe(new_bitmap.get(), is_set, 1);
        start += step;
      }

      auto *pma = new PandasMaskArray(std::move(new_bitmap));
      nb::handle py_type = nb::type<PandasMaskArray>();
      return nb::inst_take_ownership(py_type, pma);
    }

    throw nb::type_error("Invalid data type for GetItem");
  }

  auto SetItem(nb::object indexer_obj, nb::object value_obj) {
    // scalar indexer
    ssize_t i;
    if (nb::try_cast(indexer_obj, i, false)) {
      bool value;
      if (nb::try_cast(value_obj, value)) {
        return pImpl_->SetItem(i, value);
      } else {
        throw nb::type_error("expected scalar value with scalar indexer");
      }
    }

    // slice
    nb::slice slice_obj;
    if (nb::try_cast(indexer_obj, slice_obj, false)) {
      const auto converted_slice = slice_obj.compute(pImpl_->Length());
      auto [start, stop, step, length] = converted_slice;

      // assign scalar
      bool value;
      if (nb::try_cast(value_obj, value)) {
        // optimization for an empty slice with a scalar value assignment
        if ((start == 0) && (stop == pImpl_->Length()) && (step == 1)) {
          ArrowBitsSetTo(pImpl_->bitmap_->buffer.data, 0, stop, value);
          return;
        } else {
          for (size_t i = 0; i < length; ++i) {
            if (value) {
              ArrowBitSet(pImpl_->bitmap_->buffer.data, start);
            } else {
              ArrowBitClear(pImpl_->bitmap_->buffer.data, start);
            }
            start += step;
          }
          return;
        }
      }

      // empty slice with another mask
      if (nb::isinstance<PandasMaskArray>(value_obj)) {
        const auto other = nb::cast<PandasMaskArray &>(value_obj);
        pImpl_ = std::make_unique<PandasMaskArrayImpl>(other.pImpl_->Copy());
        return;
      }
    }

    // Indexing ndarray with boolean scalar assignment
    nb::ndarray<const ssize_t, nb::ndim<1>> indices;
    if (nb::try_cast(indexer_obj, indices, false)) {
      bool value;
      if (nb::try_cast(value_obj, value)) {
        const auto vw = indices.view();
        for (ssize_t idx = 0; idx < static_cast<ssize_t>(vw.shape(0)); ++idx) {
          const auto assign_idx = indices(idx);

          if ((assign_idx < 0) || (assign_idx >= pImpl_->Length())) {
            std::stringstream ss;
            ss << "Index value out of range: " << assign_idx;
            throw std::out_of_range(ss.str());
          }

          if (value) {
            ArrowBitSet(pImpl_->bitmap_->buffer.data, assign_idx);
          } else {
            ArrowBitClear(pImpl_->bitmap_->buffer.data, assign_idx);
          }
        }
        return;
      }
    }

    // Boolean ndarray
    np_arr_type bools;
    if (nb::try_cast(indexer_obj, bools, false)) {
      // can use a fast path if the size of the indexer matches our bitmask
      const auto vw = bools.view();

      // TODO: nanoarrow has ArrowBitsUnpackInt8 but not Pack equivalent,
      // leaving some performance on the table
      // if (pImpl_->Length() == static_cast<ssize_t>(vw.shape(0))) {
      //  bool value;
      //  if (nb::try_cast(value_obj, value)) {
      // }
      if (pImpl_->Length() != static_cast<ssize_t>(vw.shape(0))) {
        throw nb::value_error(
            "__setitem__ requires indexer must be same length as bitmask");
      }

      bool value;
      if (nb::try_cast(value_obj, value)) {
        for (ssize_t idx = 0; idx < static_cast<ssize_t>(vw.shape(0)); ++idx) {
          const auto should_change = vw(idx);
          if (!should_change) {
            continue;
          } else {
            if (value) {
              ArrowBitSet(pImpl_->bitmap_->buffer.data, idx);
            } else {
              ArrowBitClear(pImpl_->bitmap_->buffer.data, idx);
            }
          }
        }
        return;
      }
    }

    // TODO: there are probably many more __setitem__ operations needed to
    // mirror NumPy
    // we can either try to implement them here or just fallback to a slow path
    // where we convert to a NumPy array
    throw nb::type_error(
        "Combination of indexer and value not implemented by pandas_mask");
  }

  template <typename OP> auto BinOp(nb::object other) const {
    np_arr_type bools;

    // ndarray
    if (nb::try_cast(other, bools, false)) {
      PandasMaskArray other_pma{bools};
      return PandasMaskArray(pImpl_->BinaryOp(*other_pma.pImpl_.get(), OP()));
    }

    if (nb::inst_check(other)) {
      const auto other_pma = nb::inst_ptr<PandasMaskArray>(other);
      return PandasMaskArray(pImpl_->BinaryOp(*other_pma->pImpl_.get(), OP()));
    }

    throw nb::type_error("Invalid other argument");
  }

  auto Bytes() const {
    auto py_bytearray = PyByteArray_FromStringAndSize(
        reinterpret_cast<const char *>(pImpl_->bitmap_->buffer.data),
        pImpl_->bitmap_->buffer.size_bytes);
    auto bytearray = nb::steal(py_bytearray);
    auto py_bytes = PyBytes_FromObject(bytearray.ptr());
    return nb::steal(py_bytes);
  }

  auto NdArray(nb::object, bool) const noexcept -> np_arr_type {
    // TODO: right now we just ignore args and kwargs, but maybe we shouldn't?
    const auto nelems = pImpl_->Length();
    bool *data = new bool[nelems];
    ArrowBitsUnpackInt8(pImpl_->bitmap_->buffer.data, 0, nelems,
                        reinterpret_cast<int8_t *>(data));
    nb::capsule owner(data, [](void *p) noexcept { delete[] (bool *)p; });

    size_t shape[1] = {static_cast<size_t>(nelems)};
    return np_arr_type(data, 1, shape, owner);
  }

  auto Shape() const noexcept { return nb::make_tuple(pImpl_->Length()); }

  auto View(const std::string &dtype) const -> np_arr_type {
    if (dtype == std::string("uint8")) {
      const size_t nbits = pImpl_->bitmap_->size_bits;
      bool *data = new bool[nbits];
      ArrowBitsUnpackInt8(pImpl_->bitmap_->buffer.data, 0, nbits,
                          reinterpret_cast<int8_t *>(data));
      nb::capsule owner(data, [](void *p) noexcept { delete[] (bool *)p; });

      size_t shape[1] = {static_cast<size_t>(nbits)};
      return np_arr_type(data, 1, shape, owner);
    }

    std::stringstream ss{};
    ss << "Invalid dtype argument: '" << dtype << "'";
    throw nb::value_error(ss.str().c_str());
  }
};

NB_MODULE(pandas_mask, m) {
  nb::class_<PandasMaskArray>(m, "PandasMaskArray")
      .def(nb::init<np_arr_type>())
      .def(nb::init<PandasMaskArray>())
      .def("__len__",
           [](const PandasMaskArray &bma) noexcept {
             return bma.pImpl_->Length();
           })
      .def("__setitem__", &PandasMaskArray::SetItem)
      .def("__getitem__", &PandasMaskArray::GetItem)
      .def("__invert__",
           [](const PandasMaskArray &bma) noexcept {
             return PandasMaskArray(bma.pImpl_->Invert());
           })
      .def("__and__", &PandasMaskArray::BinOp<std::bit_and<>>)
      .def("__or__", &PandasMaskArray::BinOp<std::bit_or<>>)
      .def("__xor__", &PandasMaskArray::BinOp<std::bit_xor<>>)
      .def("__getstate__",
           [](const PandasMaskArray &bma) {
             return bma.NdArray(nb::none(), false);
           })
      .def("__setstate__",
           [](PandasMaskArray &bma, const np_arr_type &state) {
             new (&bma) PandasMaskArray(state);
           })
      .def("__iter__",
           [](const PandasMaskArray &bma) {
             return nb::make_iterator(nb::type<PandasMaskArray>(),
                                      "value_iterator", bma.pImpl_->begin(),
                                      bma.pImpl_->end());
           })
      //.def("concatenate",
      .def_prop_ro("size",
                   [](const PandasMaskArray &bma) noexcept {
                     return bma.pImpl_->Size();
                   })
      .def_prop_ro("nbytes",
                   [](const PandasMaskArray &bma) noexcept {
                     return bma.pImpl_->NBytes();
                   })
      .def_prop_ro("bytes", &PandasMaskArray::Bytes)
      .def_prop_ro("shape", &PandasMaskArray::Shape)
      .def_prop_ro("dtype",
                   [](const PandasMaskArray &) noexcept { return "bool"; })
      .def("any", [](const PandasMaskArray &bma,
                     nb::kwargs kwargs) noexcept { return bma.pImpl_->Any(); })
      .def(
          "all",
          [](const PandasMaskArray &bma) noexcept { return bma.pImpl_->All(); })
      .def(
          "sum",
          [](const PandasMaskArray &bma) noexcept { return bma.pImpl_->Sum(); })
      //.def("take_1d", &PandasMaskArray::Take1D)
      .def("copy",
           [](const PandasMaskArray &bma) noexcept {
             return PandasMaskArray(bma.pImpl_->Copy());
           })
      .def("__array__", &PandasMaskArray::NdArray, "dtype"_a = nb::none(),
           "copy"_a = false)
      .def("view", &PandasMaskArray::View, nb::arg("dtype"))
      .def("argmin",
           [](const PandasMaskArray &bma) { return bma.pImpl_->ArgMin(); })
      .def("argmax",
           [](const PandasMaskArray &bma) { return bma.pImpl_->ArgMax(); });
}

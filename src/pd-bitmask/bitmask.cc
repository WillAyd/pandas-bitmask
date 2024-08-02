#include "bitmask_impl.h"
#include "nanoarrow.h"

#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <functional>

namespace nb = nanobind;
using np_arr_type = nb::ndarray<nb::numpy, uint8_t, nb::shape<-1>>;

class BitmaskArray {
public:
  // We use a pImpl for anything that can be implemented without
  // the Python runtime. This makes it easier to test at a low level
  // and use tools like ASAN
  std::unique_ptr<BitmaskArrayImpl> pImpl_;

  explicit BitmaskArray(BitmaskArrayImpl &&bmi)
      : pImpl_(std::make_unique<BitmaskArrayImpl>(std::move(bmi))) {}

  explicit BitmaskArray(np_arr_type np_array) {
    pImpl_ = std::make_unique<BitmaskArrayImpl>(BitmaskArrayImpl());
    const auto vw = np_array.view();
    const auto nelems = vw.shape(0);

    ArrowBitmapInit(pImpl_->bitmap_.get());
    NANOARROW_THROW_NOT_OK(ArrowBitmapReserve(pImpl_->bitmap_.get(), nelems));
    ArrowBitmapAppendInt8Unsafe(
        pImpl_->bitmap_.get(),
        reinterpret_cast<const int8_t *>(np_array.data()), nelems);
  }

  explicit BitmaskArray(nanoarrow::UniqueBitmap &&bitmap)
      : pImpl_(std::make_unique<BitmaskArrayImpl>(
            BitmaskArrayImpl(std::move(bitmap)))) {}

  auto Bytes() const {
    auto py_bytearray = PyByteArray_FromStringAndSize(
        reinterpret_cast<const char *>(pImpl_->bitmap_->buffer.data),
        pImpl_->bitmap_->buffer.size_bytes);
    auto bytearray = nb::steal(py_bytearray);
    auto py_bytes = PyBytes_FromObject(bytearray.ptr());
    return nb::steal(py_bytes);
  }

  auto NdArray() const noexcept -> np_arr_type {
    const auto nelems = pImpl_->Length();
    bool *data = new bool[nelems];
    ArrowBitsUnpackInt8(pImpl_->bitmap_->buffer.data, 0, nelems,
                        reinterpret_cast<int8_t *>(data));
    nb::capsule owner(data, [](void *p) noexcept { delete[] (bool *)p; });

    size_t shape[1] = {static_cast<size_t>(nelems)};
    return np_arr_type(data, 1, shape, owner);
  }
};

NB_MODULE(bitmask, m) {
  nb::class_<BitmaskArray>(m, "BitmaskArray")
      .def(nb::init<np_arr_type>())
      .def(
          "__len__",
          [](const BitmaskArray &bma) noexcept { return bma.pImpl_->Length(); })
      .def("__setitem__",
           [](const BitmaskArray &bma, Py_ssize_t index, bool value) {
             return bma.pImpl_->SetItem(index, value);
           })
      .def("__getitem__",
           [](const BitmaskArray &bma, Py_ssize_t index) {
             return bma.pImpl_->GetItem(index);
           })
      .def("__invert__",
           [](const BitmaskArray &bma) noexcept {
             return BitmaskArray(bma.pImpl_->Invert());
           })
      .def("__and__",
           [](const BitmaskArray &bma, const BitmaskArray &other) {
             return BitmaskArray(
                 bma.pImpl_->BinaryOp(*other.pImpl_.get(), std::bit_and()));
           })
      .def("__or__",
           [](const BitmaskArray &bma, const BitmaskArray &other) {
             return BitmaskArray(
                 bma.pImpl_->BinaryOp(*other.pImpl_.get(), std::bit_or()));
           })
      .def("__xor__",
           [](const BitmaskArray &bma, const BitmaskArray &other) {
             return BitmaskArray(
                 bma.pImpl_->BinaryOp(*other.pImpl_.get(), std::bit_xor()));
           })
      .def("__getstate__",
           [](const BitmaskArray &bma) { return bma.NdArray(); })
      .def("__setstate__",
           [](BitmaskArray &bma, const np_arr_type &state) {
             new (&bma) BitmaskArray(state);
           })
      .def("__iter__",
           [](const BitmaskArray &bma) {
             return nb::make_iterator(nb::type<BitmaskArray>(),
                                      "value_iterator", bma.pImpl_->begin(),
                                      bma.pImpl_->end());
           })
      //.def("concatenate",
      .def_prop_ro(
          "size",
          [](const BitmaskArray &bma) noexcept { return bma.pImpl_->Size(); })
      .def_prop_ro(
          "nbytes",
          [](const BitmaskArray &bma) noexcept { return bma.pImpl_->NBytes(); })
      .def_prop_ro("bytes", &BitmaskArray::Bytes)
      //.def_prop_ro("shape", &BitmaskArray::Shape)
      .def_prop_ro("dtype",
                   [](const BitmaskArray &bma) noexcept { return "bool"; })
      .def("any",
           [](const BitmaskArray &bma) noexcept { return bma.pImpl_->Any(); })
      .def("all",
           [](const BitmaskArray &bma) noexcept { return bma.pImpl_->All(); })
      .def("sum",
           [](const BitmaskArray &bma) noexcept { return bma.pImpl_->Sum(); })
      //.def("take_1d", &BitmaskArray::Take1D)
      .def("copy",
           [](const BitmaskArray &bma) noexcept {
             return BitmaskArray(bma.pImpl_->Copy());
           })
      .def("__array__", &BitmaskArray::NdArray);
}

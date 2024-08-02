#include "bitmask_impl.h"
#include "nanoarrow.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <functional>

namespace nb = nanobind;

class BitmaskArray {
public:
  explicit BitmaskArray(nb::ndarray<uint8_t, nb::shape<-1>> np_array) {
    impl_ = std::make_unique<BitmaskArrayImpl>(BitmaskArrayImpl());
    const auto vw = np_array.view();
    const auto nelems = vw.shape(0);

    ArrowBitmapInit(impl_->bitmap_.get());
    NANOARROW_THROW_NOT_OK(ArrowBitmapReserve(impl_->bitmap_.get(), nelems));
    ArrowBitmapAppendInt8Unsafe(
        impl_->bitmap_.get(), reinterpret_cast<const int8_t *>(np_array.data()),
        nelems);
  }

  explicit BitmaskArray(nanoarrow::UniqueBitmap &&bitmap)
      : impl_(std::make_unique<BitmaskArrayImpl>(
            BitmaskArrayImpl(std::move(bitmap)))) {}

  auto Length() const noexcept -> Py_ssize_t { return impl_->Length(); }
  auto SetItem(Py_ssize_t index, bool value) const -> void {
    impl_->SetItem(index, value);
  }
  auto GetItem(Py_ssize_t index) const -> bool { return impl_->GetItem(index); }
  auto Invert() const noexcept -> BitmaskArray {
    return BitmaskArray(impl_->Invert());
  }
  auto And(const BitmaskArray &other) const -> BitmaskArray {
    return BitmaskArray(impl_->BinaryOp(*other.impl_.get(), std::bit_and()));
  }
  auto Or(const BitmaskArray &other) const -> BitmaskArray {
    return BitmaskArray(impl_->BinaryOp(*other.impl_.get(), std::bit_or()));
  }

  auto XOr(const BitmaskArray &other) const -> BitmaskArray {
    return BitmaskArray(impl_->BinaryOp(*other.impl_.get(), std::bit_xor()));
  }

  auto Size() const noexcept -> Py_ssize_t { return impl_->Size(); }
  auto NBytes() const noexcept -> Py_ssize_t { return impl_->NBytes(); }

  auto Bytes() const {
    auto py_bytearray = PyByteArray_FromStringAndSize(
        reinterpret_cast<const char *>(impl_->bitmap_->buffer.data),
        impl_->bitmap_->buffer.size_bytes);
    auto bytearray = nb::steal(py_bytearray);
    auto py_bytes = PyBytes_FromObject(bytearray.ptr());
    return nb::steal(py_bytes);
  }

  auto DType() const noexcept -> const char * { return "bool"; }
  auto Any() const noexcept -> bool { return impl_->Any(); }
  auto All() const noexcept -> bool { return impl_->All(); }
  auto Sum() const noexcept -> Py_ssize_t { return impl_->Sum(); }

  auto Copy() const noexcept -> BitmaskArray {
    return BitmaskArray(impl_->Copy());
  }

  auto NdArray() const noexcept
      -> nb::ndarray<nb::numpy, const bool, nb::ndim<1>> {
    const auto nelems = this->Length();
    bool *data = new bool[nelems];
    ArrowBitsUnpackInt8(impl_->bitmap_->buffer.data, 0, nelems,
                        reinterpret_cast<int8_t *>(data));
    nb::capsule owner(data, [](void *p) noexcept { delete[] (bool *)p; });

    size_t shape[1] = {static_cast<size_t>(nelems)};
    return nb::ndarray<nb::numpy, const bool, nb::ndim<1>>(data, 1, shape,
                                                           owner);
  }

private:
  explicit BitmaskArray(BitmaskArrayImpl &&bmi)
      : impl_(std::make_unique<BitmaskArrayImpl>(std::move(bmi))) {}

  std::unique_ptr<BitmaskArrayImpl> impl_;
};

NB_MODULE(bitmask, m) {
  nb::class_<BitmaskArray>(m, "BitmaskArray")
      .def(nb::init<nb::ndarray<uint8_t, nb::shape<-1>>>())
      .def("__len__", &BitmaskArray::Length)
      .def("__setitem__", &BitmaskArray::SetItem)
      .def("__getitem__", &BitmaskArray::GetItem)
      .def("__invert__", &BitmaskArray::Invert)
      .def("__and__", &BitmaskArray::And)
      .def("__or__", &BitmaskArray::Or)
      .def("__xor__", &BitmaskArray::XOr)
      //.def("__getstate__",
      //.def("__setstate__",
      //.def("__iter__",
      //.def("concatenate",
      .def_prop_ro("size", &BitmaskArray::Size)
      .def_prop_ro("nbytes", &BitmaskArray::NBytes)
      .def_prop_ro("bytes", &BitmaskArray::Bytes)
      //.def_prop_ro("shape", &BitmaskArray::Shape)
      .def_prop_ro("dtype", &BitmaskArray::DType)
      .def("any", &BitmaskArray::Any)
      .def("all", &BitmaskArray::All)
      .def("sum", &BitmaskArray::Sum)
      //.def("take_1d", &BitmaskArray::Take1D)
      .def("copy", &BitmaskArray::Copy)
      .def("__array__", &BitmaskArray::NdArray);
}

#include "bitmask_impl.h"

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
  auto SetItem(Py_ssize_t index, bool value) const -> bool {
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

  auto GetPyBuffer() const noexcept -> std::byte * {
    return impl_->ExposeBufferForPython();
  }
  auto ReleasePyBuffer() const noexcept -> void {
    return impl_->ReleasePyBuffer();
  }

private:
  explicit BitmaskArray(BitmaskArrayImpl &&bmi)
      : impl_(std::make_unique<BitmaskArrayImpl>(std::move(bmi))) {}

  std::unique_ptr<BitmaskArrayImpl> impl_;
};

int GetBuffer(PyObject *self, Py_buffer *buffer, int flags) {
  BitmaskArray *array = nb::inst_ptr<BitmaskArray>(self);
  const auto nelems = array->Length();
  buffer->buf = array->GetPyBuffer();
  buffer->format = strdup("?");
  buffer->internal = nullptr;
  buffer->itemsize = 1;
  buffer->len = nelems;
  buffer->ndim = 1; // TODO: don't hard code this
  buffer->obj = self;
  buffer->readonly = 1;
  buffer->shape = new Py_ssize_t[1]{nelems};
  buffer->strides = new Py_ssize_t[1]{1};
  buffer->suboffsets = nullptr;

  return 0;
}

void ReleaseBuffer(PyObject *self, Py_buffer *buffer) {
  delete buffer->shape;
  delete buffer->strides;

  BitmaskArray *array = nb::inst_ptr<BitmaskArray>(self);
  array->ReleasePyBuffer();
}

PyType_Slot slots[] = {{Py_bf_getbuffer, (void *)GetBuffer},
                       {Py_bf_releasebuffer, (void *)ReleaseBuffer},
                       {0, nullptr}};

NB_MODULE(bitmask, m) {
  nb::class_<BitmaskArray>(m, "BitmaskArray", nb::type_slots(slots))
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
      .def_prop_ro("size", &BitmaskArray::Size);
}

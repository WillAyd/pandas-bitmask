#include "nanoarrow.h"
#include <nanoarrow/nanoarrow.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <stdexcept>

namespace nb = nanobind;

class BitmaskArray {
public:
  // Would love for these to be private, but needed to be accessed by Py_Buffer functions
  nanoarrow::UniqueBitmap bitmap_;  
  std::byte *py_buffer;  // Python buffer protocol does not support bits

  explicit BitmaskArray(nb::ndarray<uint8_t, nb::shape<-1>> np_array) {
    const auto vw = np_array.view();
    const auto nelems = vw.shape(0);
    
    ArrowBitmapInit(bitmap_.get());
    NANOARROW_THROW_NOT_OK(ArrowBitmapReserve(bitmap_.get(), nelems));
    ArrowBitmapAppendInt8Unsafe(bitmap_.get(), reinterpret_cast<const int8_t*>(np_array.data()), nelems);
  }

  explicit BitmaskArray(nanoarrow::UniqueBitmap&& bitmap) :
    bitmap_(std::move(bitmap)) {}

  auto Length() const noexcept -> Py_ssize_t {
    return bitmap_->size_bits;
  }

  auto GetItem(Py_ssize_t index) const -> bool {
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

  auto Invert() const noexcept -> BitmaskArray {
    nanoarrow::UniqueBitmap new_bitmap;
    const size_t nbits = bitmap_->size_bits;
    size_t rem = nbits % sizeof(int64_t);
    
    ArrowBitmapInit(new_bitmap.get());
    ArrowBitmapReserve(new_bitmap.get(), nbits);

    size_t n_uint64s = nbits / sizeof(uint64_t);
    uint8_t *src = bitmap_->buffer.data;
    uint8_t *dst = new_bitmap->buffer.data;

    for (size_t i = 0; i < n_uint64s; ++i) {
      uint64_t value;
      memcpy(&value, src, sizeof(uint64_t));
      value = ~value;
      memcpy(dst, &value, sizeof(uint64_t));
      src += sizeof(uint64_t);
      dst += sizeof(uint64_t);
    }

    for (size_t i = 0; i < rem; ++i) {
      uint8_t value;
      memcpy(&value, src, sizeof(uint8_t));
      value = ~value;
      memcpy(dst, &value, sizeof(uint8_t));
      src += sizeof(uint8_t);
      dst += sizeof(uint8_t);
    }
    
    new_bitmap->size_bits = nbits;
    return BitmaskArray(std::move(new_bitmap));
  }

private:

};

int GetBuffer(PyObject *self, Py_buffer* buffer, int flags) {
  BitmaskArray *array = nb::inst_ptr<BitmaskArray>(self);
  const auto nelems = array->Length();

  array->py_buffer = new std::byte[nelems];
  ArrowBitsUnpackInt8(array->bitmap_->buffer.data,
                      0,
                      nelems,
                      reinterpret_cast<int8_t *>(array->py_buffer));

  buffer->buf = array->py_buffer;
  buffer->format = strdup("?");
  buffer->internal = nullptr;
  buffer->itemsize = 1;
  buffer->len = nelems;
  buffer->ndim = 1;  // TODO: don't hard code this
  buffer->obj = self;
  buffer->readonly = 1;
  buffer->shape = new Py_ssize_t[1]{nelems};
  buffer->strides =  new Py_ssize_t[1]{1};
  buffer->suboffsets = nullptr;

  return 0;
}

void ReleaseBuffer(PyObject *self, Py_buffer* buffer) {
  delete buffer->shape;
  delete buffer->strides;
  
  BitmaskArray *array = nb::inst_ptr<BitmaskArray>(self);
  delete array->py_buffer;
}

PyType_Slot slots[] = {
  { Py_bf_getbuffer, (void *)GetBuffer },
  { Py_bf_releasebuffer, (void *)ReleaseBuffer },
  {0, nullptr}
};

NB_MODULE(bitmap, m) {
  nb::class_<BitmaskArray>(m, "BitmaskArray", nb::type_slots(slots))
    .def(nb::init<nb::ndarray<uint8_t, nb::shape<-1>>>())
    .def("__len__", &BitmaskArray::Length)
    .def("__getitem__", &BitmaskArray::GetItem)
    .def("__invert__", &BitmaskArray::Invert);
}

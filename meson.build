project(
    'pd-bitmask',
    'cpp',
    version: '0.0.1.dev0',
    license: 'Apache-2.0',
    default_options: [
        'warning_level=2',
        'cpp_std=c++17',
    ],
)

py = import('python').find_installation()

nanobind_dep = dependency('nanobind')
nanoarrow_dep = dependency('nanoarrow')

impl_lib = library(
    'bitmask_impl',
    ['src/pd-bitmask/bitmask_impl.cc'],
    dependencies: [nanoarrow_dep],
    install: true,
)

gtest_dep = dependency('gtest_main')
impl_test = executable(
    'bitmask-impl-test',
    sources: ['src/pd-bitmask/bitmask_impl_test.cc'],
    link_with: [impl_lib],
    dependencies: [nanoarrow_dep, gtest_dep],
)
test('bitmask-impl', impl_test)

py.extension_module(
    'bitmask',
    ['src/pd-bitmask/bitmask.cc'],
    link_with: impl_lib,
    dependencies: [nanobind_dep, nanoarrow_dep],
    install: true,
)

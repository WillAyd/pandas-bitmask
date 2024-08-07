project(
    'pandas-mask',
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
nanoarrow_dep = dependency('nanoarrow', default_options: ['default_library=static'])

impl_dep = declare_dependency(
    sources: ['src/pandas-mask/pandas_mask_impl.cc'],
    dependencies: [nanoarrow_dep],
)

gtest_dep = dependency('gtest_main')
impl_test = executable(
    'pandas-mask-impl-test',
    sources: ['src/pandas-mask/pandas_mask_impl_test.cc'],
    dependencies: [gtest_dep, impl_dep],
)
test('pandas-mask-impl', impl_test)

py.extension_module(
    'pandas_mask',
    ['src/pandas-mask/pandas_mask.cc'],
    dependencies: [nanobind_dep, impl_dep],
    install: true,
)

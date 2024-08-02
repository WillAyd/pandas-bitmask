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

impl_dep = declare_dependency(
    sources: ['src/pd-bitmask/bitmask_impl.cc'],
    dependencies: [nanoarrow_dep],
)

gtest_dep = dependency('gtest_main')
impl_test = executable(
    'bitmask-impl-test',
    sources: ['src/pd-bitmask/bitmask_impl_test.cc'],
    dependencies: [gtest_dep, impl_dep],
)
test('bitmask-impl', impl_test)

py.extension_module(
    'bitmask',
    ['src/pd-bitmask/bitmask.cc'],
    dependencies: [nanobind_dep, impl_dep],
    install: true,
)

project(
    'pd-bitmap',
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

py.extension_module(
    'bitmap',
    ['src/pd-bitmap/bitmap.cc'],
    dependencies: [nanobind_dep, nanoarrow_dep],
    install: true,
)

load("@com_grail_bazel_compdb//:defs.bzl", "compilation_database")
load("@com_grail_bazel_output_base_util//:defs.bzl", "OUTPUT_BASE")

# Target to generate a compile_commands.json compilation database file
compilation_database(
    name = "compdb",
    output_base = OUTPUT_BASE,
    targets = [
        "//:typesense-server",
        "//:search",
        "//:benchmark",
    ],
)

filegroup(
    name = "src_files",
    srcs = glob(["src/*.cpp"]),
)

cc_library(
    name = "headers",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
    ]),
    includes = ["include"],
)

cc_library(
    name = "common_deps",
    defines = [
        "NDEBUG",
    ],
    linkopts = select({
        "@platforms//os:macos": ["-framework Foundation -framework SystemConfiguration"],
        "//conditions:default": [],
    }),
    deps = [
        ":headers",
        ":onnxruntime_lib",
        "@com_github_brpc_braft//:braft",
        "@com_github_brpc_brpc//:brpc",
        "@com_github_google_glog//:glog",
        "@curl",
        "@for",
        "@h2o",
        "@iconv",
        "@icu",
        "@jemalloc",
        "@kakasi",
        "@lrucache",
        "@rocksdb",
        "@s2geometry",
        "@hnsw",
    ],
)

cc_library(
    name = "linux_deps",
    defines = [
        "NDEBUG",
    ],
    deps = [
        "@elfutils//:libdw",
    ],
)

COPTS = [
    "-Wall",
    "-Wextra",
    "-Wno-unused-parameter",
    "-Werror=return-type",
    "-O2",
    "-g",
]

cc_binary(
    name = "typesense-server",
    srcs = [
        "src/main/typesense_server.cpp",
        ":src_files",
    ],
    local_defines = [
        "TYPESENSE_VERSION=\\\"$(TYPESENSE_VERSION)\\\""
    ],
    linkopts = select({
        "@platforms//os:linux": ["-static-libstdc++", "-static-libgcc", "-fsanitize=address"],
    }),
    copts = COPTS + select({
        "@platforms//os:linux": ["-DBACKWARD_HAS_DW=1", "-DBACKWARD_HAS_UNWIND=1", "-DASAN_BUILD",
                                 "-fsanitize=address", "-fno-omit-frame-pointer"],
    }),
    deps = [":common_deps"] +  select({
        "@platforms//os:linux": [":linux_deps"],
    }),
)

cc_binary(
    name = "search",
    srcs = [
        "src/main/main.cpp",
        ":src_files",
    ],
    copts = COPTS,
    deps = [":common_deps"],
)

cc_binary(
    name = "benchmark",
    srcs = [
        "src/main/benchmark.cpp",
        ":src_files",
    ],
    copts = COPTS,
    deps = [":common_deps"],
)

filegroup(
    name = "test_src_files",
    srcs = glob(["test/*.cpp"]),
)

filegroup(
    name = "test_data_files",
    srcs = glob([
        "test/**/*.txt",
        "test/**/*.ini",
        "test/**/*.jsonl",
    ]),
)

TEST_COPTS = [
    "-Wall",
    "-Wextra",
    "-Wno-unused-parameter",
    "-Werror=return-type",
    "-g",
]

ASAN_COPTS = [
    "-fsanitize=address",
    "-fno-omit-frame-pointer",
    "-DASAN_BUILD"
]

config_setting(
    name = "release_mode",
    define_values = { "mode": "release" }
)

config_setting(
    name = "asan_mode",
    define_values = { "mode": "asan" }
)

cc_test(
    name = "typesense-test",
    srcs = [
        ":src_files",
        ":test_src_files",
    ],
    copts = TEST_COPTS + select({
        ":release_mode": ["-O2"],
        ":asan_mode": ["-O0"] + ASAN_COPTS,
        "//conditions:default": ["-O0"]
    }),
    data = [
        ":test_data_files",
        "@libart//:data",
        "@token_offsets//file",
    ],
    deps = [
        ":common_deps",
        "@com_google_googletest//:gtest",
    ],
    defines = [
        "ROOT_DIR="
    ],
    linkopts = select({
       ":asan_mode": ["-fsanitize=address"],
       "//conditions:default": []
   })
)

load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

__POSTFIX = """
mkdir -p $INSTALLDIR/lib
mkdir -p $INSTALLDIR/lib/_deps
mkdir -p $INSTALLDIR/lib/_deps/onnx-build
mkdir -p $INSTALLDIR/lib/_deps/re2-build
mkdir -p  $INSTALLDIR/lib/_deps/abseil_cpp-build
mkdir -p $INSTALLDIR/lib/_deps/abseil_cpp-build/absl
mkdir -p $INSTALLDIR/lib/_deps/abseil_cpp-build/absl/base
mkdir -p $INSTALLDIR/lib/_deps/protobuf-build
mkdir -p $INSTALLDIR/lib/_deps/abseil_cpp-build/absl/container
mkdir -p $INSTALLDIR/lib/_deps/abseil_cpp-build/absl/hash
mkdir -p $INSTALLDIR/lib/_deps/pytorch_cpuinfo-build
mkdir -p $INSTALLDIR/lib/_deps/pytorch_cpuinfo-build/deps
mkdir -p $INSTALLDIR/lib/_deps/pytorch_cpuinfo-build/deps/clog
mkdir -p $INSTALLDIR/lib/_deps/google_nsync-build
cp $BUILD_TMPDIR/_deps/onnx-build/libonnx.a $INSTALLDIR/lib/_deps/onnx-build
cp $BUILD_TMPDIR/_deps/onnx-build/libonnx_proto.a $INSTALLDIR/lib/_deps/onnx-build
cp $BUILD_TMPDIR/_deps/re2-build/libre2.a $INSTALLDIR/lib/_deps/re2-build
cp $BUILD_TMPDIR/_deps/abseil_cpp-build/absl/base/libabsl_base.a $INSTALLDIR/lib/_deps/abseil_cpp-build/absl/base
cp $BUILD_TMPDIR/_deps/abseil_cpp-build/absl/base/libabsl_throw_delegate.a $INSTALLDIR/lib/_deps/abseil_cpp-build/absl/base
cp $BUILD_TMPDIR/_deps/abseil_cpp-build/absl/container/libabsl_raw_hash_set.a $INSTALLDIR/lib/_deps/abseil_cpp-build/absl/container
cp $BUILD_TMPDIR/_deps/abseil_cpp-build/absl/hash/libabsl_hash.a $INSTALLDIR/lib/_deps/abseil_cpp-build/absl/hash
cp $BUILD_TMPDIR/_deps/abseil_cpp-build/absl/hash/libabsl_city.a $INSTALLDIR/lib/_deps/abseil_cpp-build/absl/hash
cp $BUILD_TMPDIR/_deps/abseil_cpp-build/absl/hash/libabsl_low_level_hash.a $INSTALLDIR/lib/_deps/abseil_cpp-build/absl/hash
cp $BUILD_TMPDIR/_deps/google_nsync-build/libnsync_cpp.a $INSTALLDIR/lib/_deps/google_nsync-build
cp $BUILD_TMPDIR/_deps/pytorch_cpuinfo-build/deps/clog/libclog.a $INSTALLDIR/lib/_deps/pytorch_cpuinfo-build/deps/clog
cp $BUILD_TMPDIR/_deps/pytorch_cpuinfo-build/libcpuinfo.a $INSTALLDIR/lib/_deps/pytorch_cpuinfo-build
"""

cmake(
    name = "onnxruntime",
    lib_source = "@onnx_runtime//:all_srcs",
    cache_entries = {'onnxruntime_RUN_ONNX_TESTS':'OFF',
'onnxruntime_GENERATE_TEST_REPORTS':'ON',
'onnxruntime_USE_MIMALLOC':'OFF',
'onnxruntime_ENABLE_PYTHON':'OFF',
'onnxruntime_BUILD_CSHARP':'OFF',
'onnxruntime_BUILD_JAVA':'OFF',
'onnxruntime_BUILD_NODEJS':'OFF',
'onnxruntime_BUILD_OBJC':'OFF',
'onnxruntime_BUILD_SHARED_LIB':'OFF',
'onnxruntime_BUILD_APPLE_FRAMEWORK':'OFF',
'onnxruntime_USE_DNNL':'OFF',
'onnxruntime_USE_NNAPI_BUILTIN':'OFF',
'onnxruntime_USE_RKNPU':'OFF',
'onnxruntime_USE_LLVM':'OFF',
'onnxruntime_ENABLE_MICROSOFT_INTERNAL':'OFF',
'onnxruntime_USE_VITISAI':'OFF',
'onnxruntime_USE_TENSORRT':'OFF',
'onnxruntime_SKIP_AND_PERFORM_FILTERED_TENSORRT_TESTS':'ON',
'onnxruntime_USE_TENSORRT_BUILTIN_PARSER':'OFF',
'onnxruntime_TENSORRT_PLACEHOLDER_BUILDER':'OFF', 'onnxruntime_USE_TVM':'OFF',
'onnxruntime_TVM_CUDA_RUNTIME':'OFF', 'onnxruntime_TVM_USE_HASH':'OFF',
'onnxruntime_USE_MIGRAPHX':'OFF', 'onnxruntime_CROSS_COMPILING':'OFF',
'onnxruntime_DISABLE_CONTRIB_OPS':'OFF', 'onnxruntime_DISABLE_ML_OPS':'OFF',
'onnxruntime_DISABLE_RTTI':'OFF', 'onnxruntime_DISABLE_EXCEPTIONS':'OFF',
'onnxruntime_MINIMAL_BUILD':'OFF', 'onnxruntime_EXTENDED_MINIMAL_BUILD':'OFF',
'onnxruntime_MINIMAL_BUILD_CUSTOM_OPS':'OFF', 'onnxruntime_REDUCED_OPS_BUILD':'OFF',
'onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS':'OFF', 'onnxruntime_USE_DML':'OFF',
'onnxruntime_USE_WINML':'OFF', 'onnxruntime_BUILD_MS_EXPERIMENTAL_OPS':'OFF',
'onnxruntime_USE_TELEMETRY':'OFF', 'onnxruntime_ENABLE_LTO':'OFF',
'onnxruntime_USE_ACL':'OFF', 'onnxruntime_USE_ACL_1902':'OFF',
'onnxruntime_USE_ACL_1905':'OFF', 'onnxruntime_USE_ACL_1908':'OFF',
'onnxruntime_USE_ACL_2002':'OFF', 'onnxruntime_USE_ARMNN':'OFF',
'onnxruntime_ARMNN_RELU_USE_CPU':'ON', 'onnxruntime_ARMNN_BN_USE_CPU':'ON',
'onnxruntime_ENABLE_NVTX_PROFILE':'OFF', 'onnxruntime_ENABLE_TRAINING':'OFF',
'onnxruntime_ENABLE_TRAINING_OPS':'OFF', 'onnxruntime_ENABLE_TRAINING_APIS':'OFF',
'onnxruntime_ENABLE_CPU_FP16_OPS':'OFF', 'onnxruntime_USE_NCCL':'OFF',
'onnxruntime_BUILD_BENCHMARKS':'OFF', 'onnxruntime_USE_ROCM':'OFF',
'Onnxruntime_GCOV_COVERAGE':'OFF', 'onnxruntime_USE_MPI':'ON',
'onnxruntime_ENABLE_MEMORY_PROFILE':'OFF',
'onnxruntime_ENABLE_CUDA_LINE_NUMBER_INFO':'OFF',
'onnxruntime_BUILD_WEBASSEMBLY':'OFF', 'onnxruntime_BUILD_WEBASSEMBLY_STATIC_LIB':'OFF',
'onnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_CATCHING':'ON',
'onnxruntime_ENABLE_WEBASSEMBLY_API_EXCEPTION_CATCHING':'OFF',
'onnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_THROWING':'ON',
'onnxruntime_ENABLE_WEBASSEMBLY_THREADS':'OFF',
'onnxruntime_ENABLE_WEBASSEMBLY_DEBUG_INFO':'OFF',
'onnxruntime_ENABLE_WEBASSEMBLY_PROFILING':'OFF',
'onnxruntime_ENABLE_EAGER_MODE':'OFF', 'onnxruntime_ENABLE_LAZY_TENSOR':'OFF',
'onnxruntime_ENABLE_EXTERNAL_CUSTOM_OP_SCHEMAS':'OFF', 'onnxruntime_ENABLE_CUDA_PROFILING':'OFF',
'onnxruntime_ENABLE_ROCM_PROFILING':'OFF', 'onnxruntime_USE_XNNPACK':'OFF',
'onnxruntime_USE_CANN':'OFF', 'CMAKE_TLS_VERIFY':'ON', 'FETCHCONTENT_QUIET':'OFF',
'onnxruntime_PYBIND_EXPORT_OPSCHEMA':'OFF', 'onnxruntime_ENABLE_MEMLEAK_CHECKER':'OFF',
'CMAKE_BUILD_TYPE':'Release',
},
    working_directory="cmake",
    build_args= [
        "--config Release",
        "-j6"
    ],
    tags=["requires-network","no-sandbox"],
    features=["-default_compile_flags","-fno-canonical-system-headers", "-Wno-builtin-macro-redefined"],
    out_static_libs=[ "libonnxruntime_session.a",
    "libonnxruntime_optimizer.a",
    "libonnxruntime_providers.a",
    "libonnxruntime_util.a",
    "libonnxruntime_framework.a",
    "libonnxruntime_graph.a",
    "libonnxruntime_mlas.a",
    "libonnxruntime_common.a",
    "libonnxruntime_flatbuffers.a",
    "_deps/onnx-build/libonnx.a",
    "_deps/onnx-build/libonnx_proto.a",
    "_deps/re2-build/libre2.a",
    "_deps/abseil_cpp-build/absl/base/libabsl_base.a",
    "_deps/abseil_cpp-build/absl/base/libabsl_throw_delegate.a",
    "_deps/abseil_cpp-build/absl/container/libabsl_raw_hash_set.a",
    "_deps/abseil_cpp-build/absl/hash/libabsl_hash.a",
    "_deps/abseil_cpp-build/absl/hash/libabsl_city.a",
    "_deps/abseil_cpp-build/absl/hash/libabsl_low_level_hash.a",
    "_deps/google_nsync-build/libnsync_cpp.a",
    "_deps/pytorch_cpuinfo-build/libcpuinfo.a",
    "_deps/pytorch_cpuinfo-build/deps/clog/libclog.a",
    ],
    postfix_script=__POSTFIX
)

cc_library(
    name = "onnxruntime_lib",
    linkopts = select({
        "@platforms//os:linux": ["-static-libstdc++", "-static-libgcc"],
        "//conditions:default": [],
    }),
    deps = ["//:onnxruntime", "@onnx_runtime_extensions//:operators"],
    includes= ["onnxruntime/include/onnxruntime"]
)

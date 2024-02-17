load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "hdr_histogram",
    srcs = glob([
        "@hdr_histogram//src/*.c",  # Adjust the path to match HDRHistogram's source files
    ]),
    hdrs = glob([
        "@hdr_histogram//src/*.h",  # Adjust the path to match HDRHistogram's header files
    ]),
    includes = ["@hdr_histogram//src"],
    visibility = ["//visibility:public"],
)

###
# Copyright (C) 2023 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

KANN_POCL_MAKEFILE ?= $(KALRAY_TOOLCHAIN_DIR)/share/make/kann/kann_runtime.mk

O := output
cluster-arch := kv3-2
cluster-system := cos
use-module := strict-flags

all: mppa-kann-custom

mppa-kann-custom-name := mppa-kann-custom
mppa-kann-custom-cflags := -g -Wall -Wextra -Werror -std=gnu11 -O3 -fipa-pta
mppa-kann-custom-cflags += -ffunction-sections -fdata-sections -fno-common -ffast-math
mppa-kann-custom-cflags += -lpthread
mppa-kann-custom-cflags += -Wa,--check-resources
mppa-kann-custom-cflags += -fPIC
mppa-kann-custom-cflags += -D NDEBUG -save-temps
mppa-kann-custom-cflags += -falign-functions=64
mppa-kann-custom-cflags += -I./kernels/include
mppa-kann-custom-cflags += -Wno-error=unused-parameter
mppa-kann-custom-srcs := $(wildcard kernels/*.c)
cluster-lib += mppa-kann-custom


include $(KANN_POCL_MAKEFILE)
include $(KALRAY_TOOLCHAIN_DIR)/share/make/Makefile.kalray

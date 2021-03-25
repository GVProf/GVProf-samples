SHELL=/bin/sh -ue

CFLAGS   += -O3 -g
CXXFLAGS += -O3

ifdef OUTPUT
CPPFLAGS += -DOUTPUT
endif

ifdef DEBUG
CFLAGS   += -g
CXXFLAGS += -g
endif

# include Make.user relative to every active Makefile, exactly once
MAKEFILE_DIRS = $(foreach MAKEFILE,$(realpath $(MAKEFILE_LIST)), $(shell dirname $(MAKEFILE)))
$(foreach DIR,$(sort $(MAKEFILE_DIRS)),\
	$(eval -include $(DIR)/Make.user)\
)



#
# Auxiliary
#

DUMMY=
SPACE=$(DUMMY) $(DUMMY)
COMMA=$(DUMMY),$(DUMMY)

define join-list
$(subst $(SPACE),$(2),$(1))
endef


#
# CUDA detection
#

NVCC=nvcc

LDLIBS  += -lcudart -lnvToolsExt


#
# NVCC compilation
#

# NOTE: passing -lcuda to nvcc is redundant, and shouldn't happen via -Xcompiler
# TODO: pass all CXXFLAGS to nvcc using -Xcompiler (i.e. -O3, -g, etc.)
NONCUDA_LDLIBS = $(filter-out -lcuda -lcudart,$(LDLIBS))


ifneq ($(strip $(NONCUDA_LDLIBS)),)
NVCC_LDLIBS += -Xcompiler $(call join-list,$(NONCUDA_LDLIBS),$(COMMA))
endif
NVCC_LDLIBS += -lcuda -lnvToolsExt

GPU_ARCH ?=

NVCCFLAGS += --generate-line-info $(GPU_ARCH) -g -O3
ifdef DEBUG
NVCCFLAGS += -g --device-debug
endif

%: %.cu
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $^

%.o: %.cu
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c -o $@ $<

%.ptx: %.cu
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -ptx -o $@ $<

GOOGLE_TEST = tests/googletest
GOOGLE_TEST_BUILD = $(GOOGLE_TEST)/build
NVCC=/usr/local/cuda/bin/nvcc
TEST_INCLUDE_DIRS = -Isrc/include/ -I$(GOOGLE_TEST)/googletest/include/ -L$(GOOGLE_TEST_BUILD)/lib/
TEST_LFLAGS = -lgtest -lpthread
GOOGLE_TEST_MAIN = $(GOOGLE_TEST)/googletest/src/gtest_main.cc

tests: build-googletest

build-googletest: $(GOOGLE_TEST)
	mkdir -p $(GOOGLE_TEST_BUILD) && cd $(GOOGLE_TEST_BUILD) && cmake .. && make -j

simple-test: build-googletest
	$(NVCC) tests/$@.cu $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -o $@ -g
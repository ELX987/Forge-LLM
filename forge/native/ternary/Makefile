# tern158 standalone build (no PyTorch needed):   make selftest && ./build/tern158_selftest --bench
NVCC    ?= nvcc
PYTHON  ?= python3
GENCODE ?= $(shell $(PYTHON) tools/detect_arch.py --gencode)
CXXFLAGS = -O3 -std=c++17 --expt-relaxed-constexpr -lineinfo -Xcompiler -fPIC -Ikernel/include
SRC      = $(wildcard kernel/src/*.cu)
OBJ      = $(patsubst kernel/src/%.cu,build/%.o,$(SRC))

all: build/libtern158.so build/tern158_selftest

build/%.o: kernel/src/%.cu $(wildcard kernel/include/tern158/*)
	@mkdir -p build
	$(NVCC) $(CXXFLAGS) $(GENCODE) -c $< -o $@

build/libtern158.so: $(OBJ)
	$(NVCC) -shared $(GENCODE) $^ -o $@ -lcublas

build/tern158_selftest: kernel/tests/selftest.cu $(OBJ)
	$(NVCC) $(CXXFLAGS) $(GENCODE) $^ -o $@ -lcublas

selftest: build/tern158_selftest
	./build/tern158_selftest --bench --json build/selftest.json

syntax:
	$(PYTHON) tools/syntax_check.py

clean:
	rm -rf build

.PHONY: all selftest syntax clean

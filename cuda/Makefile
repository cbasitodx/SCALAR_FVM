NVCC      = nvcc
CC        = gcc
# -O2: compiler optimizations (needed for meaningful timings)
NVCCFLAGS = -O2 -I. -I../common -arch=sm_75
CFLAGS    = -O2 -Wall -Wextra -I../common
LIBS      = -lm

# Folders where make looks for source files
# (the solver in src/*.inl is not compiled on its own: it is included through include/*.cuh)
vpath %.cu tests            # test programs
vpath %.c  ../common/src    # shared code (HistoryBinary.c)

# One executable per file in tests/ (tests/testBurgersCUDA.cu -> testBurgersCUDA)
TESTS  = $(patsubst tests/%.cu,%,$(wildcard tests/*.cu))
COMMON = obj/HistoryBinary.o

.PHONY: all run clean cleanbin

all: $(TESTS)

# Build and run every test
run: $(TESTS)
	for t in $(TESTS); do echo "== $$t"; ./$$t || exit 1; done

# Link: each test = its own object + the shared objects
$(TESTS): %: obj/%.o $(COMMON)
	$(NVCC) $^ -o $@ $(LIBS)

# Compile: any X.cu (found through vpath) -> obj/X.o with nvcc. Recompile if a header or the solver changes
obj/%.o: %.cu $(wildcard include/*.cuh src/*.inl ../common/include/*.h)
	mkdir -p obj
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile: any X.c (found through vpath) -> obj/X.o as C with gcc
obj/%.o: %.c $(wildcard ../common/include/*.h)
	mkdir -p obj
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf obj $(TESTS)

# Remove the binary history files written by the tests (in this folder)
cleanbin:
	rm -f *.bin

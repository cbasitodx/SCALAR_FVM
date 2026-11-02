CC     = gcc
# -O2: compiler optimizations (needed for meaningful timings)
CFLAGS = -O2 -Wall -Wextra -I. -I../../common
LIBS   = -lm

# Folders where make looks for .c files
# (the solver in src/*.inl is not compiled on its own: it is included through include/*.h)
vpath %.c tests              # test programs
vpath %.c ../../common/src   # shared code (HistoryBinary.c)

# One executable per file in tests/ (tests/testBurgers.c -> testBurgers)
TESTS  = $(patsubst tests/%.c,%,$(wildcard tests/*.c))
COMMON = obj/HistoryBinary.o

.PHONY: all run clean cleanbin

all: $(TESTS)

# Build and run every test
run: $(TESTS)
	for t in $(TESTS); do echo "== $$t"; ./$$t || exit 1; done

# Link: each test = its own object + the shared objects
$(TESTS): %: obj/%.o $(COMMON)
	$(CC) $^ -o $@ $(LIBS)

# Compile: any X.c (found through vpath) -> obj/X.o. Recompile if a header or the solver changes
obj/%.o: %.c $(wildcard include/*.h src/*.inl ../../common/include/*.h)
	mkdir -p obj
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf obj $(TESTS)

# Remove the binary history files written by the tests (in this folder)
cleanbin:
	rm -f *.bin

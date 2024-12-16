CC        = icx -std=c99 -Wall -Wno-unused-variable
CPPOPTS   = -DNTRIALS=1000 -DVERBOSE
COPTS     = -O3 -fno-alias -qopenmp -qmkl -xCORE-AVX512 -qopt-zmm-usage=high
COPTS    += -qopt-prefetch=5 -qopt-prefetch-distance=128
LOPTS     = -O3 -qmkl -qopenmp -z noexecstack

all: ref.bin intrin.bin asm.bin

bench.o: bench.c
	$(CC) -c $(CPPOPTS) $(COPTS) -o $@ $<
multi_vector_ref.o: multi_vector.c
	$(CC) -c $(CPPOPTS) $(COPTS) -o $@ $<
multi_vector_intrin.o: multi_vector.c
	$(CC) -c $(CPPOPTS) $(COPTS) -DUSE_INTRINSICS -o $@ $<
multi_vector_asm.o: multi_vector.c
	$(CC) -c $(CPPOPTS) $(COPTS) -DUSE_ASM -o $@ $<

one_vector_ker_asm.o: one_vector_ker_asm.S
	$(CC) -c $(CPPOPTS) $(COPTS) -o $@ $<
two_vector_ker_asm.o: two_vector_ker_asm.S
	$(CC) -c $(CPPOPTS) $(COPTS) -o $@ $<
four_vector_ker_asm.o: four_vector_ker_asm.S
	$(CC) -c $(CPPOPTS) $(COPTS) -o $@ $<

ref.bin: bench.o multi_vector_ref.o
	$(CC) $(LOPTS) -o $@ $^
intrin.bin: bench.o multi_vector_intrin.o
	$(CC) $(LOPTS) -o $@ $^
asm.bin: bench.o multi_vector_asm.o one_vector_ker_asm.o two_vector_ker_asm.o four_vector_ker_asm.o
	$(CC) $(LOPTS) -o $@ $^

clean:
	rm -rf *.o *.bin

.PHONY: all clean 

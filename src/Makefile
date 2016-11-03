NVCC_FLAGS = -arch=sm_35 -Xcompiler -fPIC -lcublas_device
R_INC = ${R_HOME}/include
LIBS = -lcublas_device -lcublas
MY_LIB = dev_link
DEPS = dev_link/libmax.a
OBJS = compile_fns/mat_mult.o

#make SHLIB
libsomePackage.so: somePackageall.o
	g++ -shared somePackageall.o -o libsomPackage.so -I$(R_INC) -lR

#link everything
somePackageall.o: somePackage.o Rmat_mult.o $(DEPS) $(OBJS)
	nvcc $(NVCC_FLAGS) somePackage.o Rmat_mult.o $(OBJS) -I$(R_INC) -I. $(LIBS) -L$(MY_LIB) -lmax -o somePackageall.so

#compile wrappers
somePackage.o: somePackage.cu
	nvcc -c $(NVCC_FLAGS) somePackage.cu -I$(R_INC) -I. -o somePackage.o

Rmat_mult.o: Rmat_mult.cu
	nvcc -c $(NVCC_FLAGS) Rmat_mult.cu -I$(R_INC) -I. -o Rmat_mult.o

#host functions
compile_fns/mat_mult.o: compile_fns/mat_mult.cu
	nvcc -c $(NVCC_FLAGS) $< $(LIBS) -o $@

#libraries for device linked functions
dev_link/libmax.a: dev_link/max.o
	nvcc -lib $(NVCC_FLAGS) $(LIBS) $< -o $@

dev_link/max.o: dev_link/max.cu
	nvcc -dc $(NVCC_FLAGS) $(LIBS) $< -o $@


.PHONY: clean
clean:
	rm -f somePackage.so somePackage.o Rmat_mult.o
	rm -f dev_link/max.o dev_link/libmax.a
	rm -f compile_fns/mat_mult.o


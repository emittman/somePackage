NVCC_FLAGS = -arch=sm_35 -Xcompiler -fPIC -lcublas_device
R_INC = ${R_HOME}/include
LIBS = -lcublas_device -lcublas
MY_LIB = /dev_link
DEPS = /dev_link/libmax.a

#final SHLIB
somePackage.so: somePackage.o Rmat_mult.o mat_mult.o $(DEPS)
	nvcc -shared $(NVCC_FLAGS) somePackage.o Rmat_mult.o mat_mult.o -I$(R_INC) -I. $(LIBS) -L$(MY_LIB) -lmax -o somePackage.so

#compile wrappers
somePackage.o: somePackage.cu
	nvcc -c $(NVCC_FLAGS) somePackage.cu -I$(R_INC) -I. -o somePackage.o

Rmat_mult.o: Rmat_mult.cu
	nvcc -c $(NVCC_FLAGS) Rmat_mult.cu -I$(R_INC) -I. -o Rmat_mult.o

#host functions
mat_mult.o: mat_mult.cu
	nvcc -c $(NVCC_FLAGS) mat_mult.cu $(LIBS) -o mat_mult.o

#libraries for device linked functions
/dev_link/libmax.a: max.o
	nvcc -lib $(NVCC_FLAGS) $(LIBS) $< -o $@

/dev_link/%.o: dev_link/%.cu
	nvcc -dc $(NVCC_FLAGS) $(LIBS) $< -o $@


.PHONY: clean
clean:
	rm -f somePackage.so somePackage.o Rmat_mult.o mat_mult.o libmax.a max.o


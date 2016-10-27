NVCC_FLAGS = -arch=sm_35 -Xcompiler -fPIC -lcublas_device
R_INC = /home/emittman/src/R-3.3.1/builddir/include
LIBS = -lcublas_device -lcublas

somePackage.so: somePackage.o Rmat_mult.o mat_mult.o libmax.a
	nvcc -shared $(NVCC_FLAGS) somePackage.o Rmat_mult.o mat_mult.o -I$(R_INC) -I. -L. $(LIBS) -lmax -o somePackage.so

somePackage.o: somePackage.cu libmax.a
	nvcc -c $(NVCC_FLAGS) somePackage.cu -lmax -I$(R_INC) -I. -o somePackage.o

Rmat_mult.o: Rmat_mult.cu
	nvcc -c $(NVCC_FLAGS) Rmat_mult.cu -I$(R_INC) -I. -o Rmat_mult.o

mat_mult.o: mat_mult.cu
	nvcc -c $(NVCC_FLAGS) mat_mult.cu $(LIBS) -o mat_mult.o

libmax.a: max.o
	nvcc -lib $(NVCC_FLAGS) $(LIBS) max.o -o libmax.a

max.o: max.cu
	nvcc -dc $(NVCC_FLAGS) $(LIBS) max.cu -o max.o


.PHONY: clean
clean:
	rm somePackage.so
	rm somePackage.o
	rm Rmat_mult.o
	rm mat_mult.o
	rm libmax.a
	rm max.o


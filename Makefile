include /usr/local/conf/ElVars


#henry
#CXX = /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++
#EL_INC = -I/usr/local/include
#EL_LIB = -L/usr/local/lib -lEl
#MPI_INC = -I/usr/local/Cellar/open-mpi/1.8.4/include
#MPI_LIB = /usr/local/Cellar/open-mpi/1.8.4/lib/libmpi_cxx.dylib /usr/local/Cellar/open-mpi/1.8.4/lib/libmpi.dylib
#CXX_FLAGS = -O3 -std=c++11

#stampede
#CXX = icpc
#EL_INC = -I$(WORK)/packages/Elemental/include
#EL_LIB = -L$(WORK)/packages/Elemental/lib -lEl
#MPI_INC = -I/opt/apps/intel14/mvapich2/2.0b/include
#CXX_FLAGS = -std=c++0x
#LIBS = $(EL_LIB) $(MPI_LIB)
#INCS = $(EL_INC) $(MPI_INC)


TARGET_BIN = nystrom

all : $(TARGET_BIN)

	#$(CXX) $(CXX_FLAGS) $^ $(LIBS) $(INCS) -o $@
$(TARGET_BIN) : nystrom_main.cpp 
	$(CXX) $(EL_COMPILE_FLAGS) $< $(EL_LINK_FLAGS) $(EL_LIBS) -o $@



include $(EL_DIR)/conf/ElVars
CPP_FLAGS = -openmp -g -std=c++11  
KNN_INCS =  -I$(KNN_DIR)/generator/ \
            -I$(KNN_DIR)/include/ \
            -I$(KNN_DIR)/include/binTree/ \
            -I$(KNN_DIR)/include/direct_knn/ \
            -I$(KNN_DIR)/include/repartition/ \
            -I$(KNN_DIR)/include/lsh \
            -I$(KNN_DIR)/include/stTree \
            -I$(KNN_DIR)/include/parallelIO 
KNN_LIBS = -L$(KNN_DIR)/build -lknn -lrrot

CMD_INC = -I${KNN_DIR}/external/cmd/
CMD_LIB = -L${KNN_DIR}/build -lcmd


ALL_INCS = $(EL_LINK_FLAGS) -I./ $(KNN_INCS) $(CMD_INC) 
ALL_LIBS = -L./ $(EL_LIBS) $(KNN_LIBS) $(CMD_LIB) $(NYST_INC)

GKERNEL_OBJ = gKernel.o
GKERNEL_SRC = gaussKernel.cpp
GKERNEL_DEPS = gaussKernel.hpp kernel_inputs.hpp
NYST_OBJ = nystrom_alg.o
NYST_SRC = nystrom_alg.cpp
NYST_DEPS = nystrom_alg.hpp nystrom_utils.hpp
MAIN_BIN = nystrom.exe
MAIN_OBJ = nystrom_main.o
MAIN_SRC = nystrom_main.cpp
MAIN_DEPS = nystrom_alg.hpp nystrom_utils.hpp kernel_inputs.hpp 
ALL_OBJS = $(MAIN_OBJ) $(GKERNEL_OBJ) $(NYST_OBJ)

all : $(MAIN_BIN)

$(MAIN_BIN) : $(ALL_OBJS)
	$(CXX) $(EL_COMPILE_FLAGS) $(CPP_FLAGS) $^ $(ALL_INCS) $(ALL_LIBS) -o $(MAIN_BIN)

$(MAIN_OBJ) : $(MAIN_SRC) $(MAIN_DEPS)
	$(CXX) $(EL_COMPILE_FLAGS) $(CPP_FLAGS) -c $(MAIN_SRC) $(ALL_INCS) $(ALL_LIBS) -o $@

$(NYST_OBJ) : $(NYST_SRC) $(NYST_DEPS)
	$(CXX) $(EL_COMPILE_FLAGS) $(CPP_FLAGS) -c $(NYST_SRC) $(ALL_INCS) $(ALL_LIBS) -o $@

$(GKERNEL_OBJ) : $(GKERNEL_SRC) $(GKERNEL_DEPS)
	$(CXX) $(EL_COMPILE_FLAGS) $(CPP_FLAGS) -c $(GKERNEL_SRC) $(EL_LINK_FLAGS) $(EL_LIBS) -o $@

test_matvec.exe : matvec.cpp
	$(CXX) $(EL_COMPILE_FLAGS) matvec.cpp $(EL_LINK_FLAGS) $(EL_LIBS) -o $@

test_allocate.exe : allocate.cpp
	$(CXX) $(EL_COMPILE_FLAGS) allocate.cpp $(EL_LINK_FLAGS) $(EL_LIBS) -o $@

clean:
	rm *.o
	rm *.exe


include $(EL_DIR)/conf/ElVars
CPP_FLAGS = -O3 -std=c++11  

PSC_INC = -I$(PETSC_DIR)/include
PSC_LIB = -L$(PETSC_DIR)/lib -lpetsc

ASK_LIB = -L$(ASKIT_DIR)/build/ -laskit 
ASK_INC = -I$(ASKIT_DIR)/src/

NYST_LIB = -L$(NYST_DIR) -lnystrom
NYST_INC = -I$(NYST_DIR)

ALL_INCS =  $(EL_LINK_FLAGS) $(NYST_INC)
ALL_LIBS =  $(EL_LIBS) $(NYST_LIB)

UTIL_OBJ = nystrom_utils.o
UTIL_SRC = nystrom_utils.cpp
UTIL_DEP = nystrom_utils.hpp
ASKIT_OBJ = askit_el_utils.o
ASKIT_SRC = askit_el_utils.cpp
ASKIT_DEPS = askit_el_utils.hpp
ELPSC_BIN = el_petsc_test.exe
ELPSC_SRC = el_petsc_utils.cpp
ELPSC_DEPS = el_petsc_utils.hpp
GKERNEL_OBJ = gKernel.o
GKERNEL_SRC = gaussKernel.cpp
GKERNEL_DEPS = gaussKernel.hpp kernel_inputs.hpp
NYST_OBJ = nystrom_alg.o
NYST_SRC = nystrom_alg.cpp
NYST_DEPS = nystrom_alg.hpp nystrom_utils.hpp
NYST_SO = libnystrom.so
TEST_BIN = nystrom.exe
TEST_OBJ = nystrom_tests.o
TEST_SRC = nystrom_tests.cpp
TEST_DEPS = nystrom_alg.hpp nystrom_utils.hpp kernel_inputs.hpp



NYST_OBJS = $(GKERNEL_OBJ) $(NYST_OBJ) $(UTIL_OBJ)

all : $(NYST_SO)

$(NYST_SO) : $(NYST_OBJS)
	$(CXX) $(EL_COMPILE_FLAGS) -shared $^ $(EL_LINK_FLAGS) $(EL_LIBS) -o $@

$(NYST_OBJ) : $(NYST_SRC) $(NYST_DEPS)
	$(CXX) $(EL_COMPILE_FLAGS) -fPIC -c $< $(EL_LINK_FLAGS) $(EL_LIBS) -o $@

$(GKERNEL_OBJ) : $(GKERNEL_SRC) $(GKERNEL_DEPS)
	$(CXX) $(EL_COMPILE_FLAGS) -fPIC -c $< $(EL_LINK_FLAGS) $(EL_LIBS) -o $@

$(UTIL_OBJ) : $(UTIL_SRC) $(UTIL_DEP)
	$(CXX) $(EL_COMPILE_FLAGS) -fPIC -c $< $(EL_LINK_FLAGS) $(EL_LIBS) -o $@

tests : $(TEST_BIN)

$(TEST_BIN) : $(TEST_SRC) 
	$(CXX) $(EL_COMPILE_FLAGS) $< $(ALL_INCS) $(ALL_LIBS) -o $@

el2petsc : $(ELPSC_BIN)

$(ELPSC_BIN) : $(ELPSC_SRC) $(ELPSC_DEPS)
	$(CXX) $(EL_COMPILE_FLAGS) $(CPP_FLAGS) $(ELPSC_SRC) $(PSC_INC) $(EL_LINK_FLAGS) $(PSC_LIB) $(EL_LIBS) -o $@

askit : $(ASKIT_OBJ)

$(ASKIT_OBJ) : $(ASKIT_SRC) $(ASKIT_DEPS)
	$(CXX) $(EL_COMPILE_FLAGS) $(CPP_FLAGS) -c $(ASKIT_SRC) $(KNN_INCS) $(ASK_INC) $(EL_LINK_FLAGS) $(KNN_LIBS) $(EL_LIBS) $(ASK_LIB) -o $@

test_matvec.exe : matvec.cpp
	$(CXX) $(EL_COMPILE_FLAGS) matvec.cpp $(EL_LINK_FLAGS) $(EL_LIBS) -o $@

test_allocate.exe : allocate.cpp
	$(CXX) $(EL_COMPILE_FLAGS) allocate.cpp $(EL_LINK_FLAGS) $(EL_LIBS) -o $@

clean:
	rm *.o
	rm *.exe
	rm $(NYST_SO)


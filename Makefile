CXX = icpc

EL_INC = -I$(WORK)/packages/Elemental/include
EL_LIB = -L$(WORK)/packages/Elemental/lib -lEl

MPI_INC = -I/opt/apps/intel14/mvapich2/2.0b/include

CXX_FLAGS = -std=c++0x

LIBS = $(EL_LIB)
INCS = $(EL_INC) $(MPI_INC)

TARGET_BIN = nystrom

all : $(TARGET_BIN)

$(TARGET_BIN) : nystrom_main.cpp 
	$(CXX) $(CXX_FLAGS) $^ $(LIBS) $(INCS) -o $@



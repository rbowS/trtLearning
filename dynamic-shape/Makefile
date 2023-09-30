# Makefile for compiling CUDA files with dependencies

# Compiler and flags
NVCC := nvcc
CFLAGS := -std=c++11 -O3 -lnvinfer -lnvparsers -lnvinfer_plugin -lcudart -I/home/srb/cu12/include -I/home/srb/TensorRT-8.6.1.6/include -L/home/srb/cu12/lib64 -L/home/srb/TensorRT-8.6.1.6/lib

# Directories
SRC_DIR := ./src
OBJ_DIR := ./bin
BIN_DIR := ./bin

# List of source files
SRCS := $(wildcard $(SRC_DIR)/*.cu)
OBJS := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRCS))
DEPS := $(OBJS:.o=.d)

# Main target
MAIN := main

# Make all
all: $(BIN_DIR)/$(MAIN)

# Linking the main target
$(BIN_DIR)/$(MAIN): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(CFLAGS) $^ -o $@

# Building object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(CFLAGS) -MMD -c $< -o $@

# Include dependency files
-include $(DEPS)

# Clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

run:
	././bin/main

.PHONY: all clean








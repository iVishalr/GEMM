CC:=gcc
CFLAGS:=-O3 -msse3 -funroll-loops -I ./include/

SRC:=src
BUILD:=build
BIN:=bin
SRCS:=$(wildcard $(SRC)/*.c)
OPTIM_DIR:=$(SRC)/optimization_steps/

OBJECTS:=$(patsubst $(SRC)/%.c, $(BUILD)/%.o,$(SRCS))

all:
	@if ! test -d $(BUILD); \
		then echo "\033[93msetting up build directory...\033[0m"; mkdir -p build;\
  	fi
	@$(MAKE) gemm

gemm: $(OBJECTS)
	$(CC) $(CFLAGS) -o gemm $(OBJECTS)

optim:
	@if ! test -d $(BUILD); \
		then echo "\033[93msetting up build directory...\033[0m"; mkdir -p build;\
  	fi
	@echo "Compiling and Running Optimization Benchmark"
	@echo "This will take a while. Go grab a coffee."
	$(CC) $(CFLAGS) -o $(BUILD)/naive_matmul $(OPTIM_DIR)/naive_matmul.c
	$(CC) $(CFLAGS) -o $(BUILD)/optim1_matmul $(OPTIM_DIR)/optim1_matmul.c
	$(CC) $(CFLAGS) -o $(BUILD)/optim2_matmul $(OPTIM_DIR)/optim2_matmul.c
	$(CC) $(CFLAGS) -o $(BUILD)/optim3_matmul $(OPTIM_DIR)/optim3_matmul.c
	$(CC) $(CFLAGS) -o $(BUILD)/optim4_matmul $(OPTIM_DIR)/optim4_matmul.c
	$(CC) $(CFLAGS) -o $(BUILD)/optim5_matmul $(OPTIM_DIR)/optim5_matmul.c
	$(CC) $(CFLAGS) -o $(BUILD)/optim6_matmul $(OPTIM_DIR)/optim6_matmul.c
	$(CC) $(CFLAGS) -o $(BUILD)/optim7_matmul $(OPTIM_DIR)/optim7_matmul.c
	$(CC) $(CFLAGS) -o $(BUILD)/optim8_matmul $(OPTIM_DIR)/optim8_matmul.c
	$(CC) $(CFLAGS) -o $(BUILD)/optim9_matmul $(OPTIM_DIR)/optim9_matmul.c
	$(CC) $(CFLAGS) -o $(BUILD)/optim10_matmul $(OPTIM_DIR)/optim10_matmul.c
	$(CC) $(CFLAGS) -o $(BUILD)/optim11_matmul $(OPTIM_DIR)/optim11_matmul.c
	$(CC) $(CFLAGS) -o $(BUILD)/optim12_matmul $(OPTIM_DIR)/optim12_matmul.c
	$(CC) $(CFLAGS) -o $(BUILD)/optim13_matmul $(OPTIM_DIR)/optim13_matmul.c
	$(CC) $(CFLAGS) -o $(BUILD)/optim14_matmul $(OPTIM_DIR)/optim14_matmul.c
# I know this is stupid! :p
	$(BUILD)/naive_matmul && $(BUILD)/optim1_matmul && $(BUILD)/optim2_matmul && $(BUILD)/optim3_matmul && $(BUILD)/optim4_matmul && $(BUILD)/optim5_matmul && $(BUILD)/optim6_matmul && $(BUILD)/optim7_matmul && $(BUILD)/optim8_matmul && $(BUILD)/optim9_matmul && $(BUILD)/optim10_matmul && $(BUILD)/optim11_matmul && $(BUILD)/optim12_matmul && $(BUILD)/optim13_matmul && $(BUILD)/optim14_matmul

$(BUILD)/%.o: $(SRC)/%.c 
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD)/* $(BUILD) gemm
################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/Main.cpp \
../src/Net.cpp \
../src/Neuron.cpp \
../src/TrainingData.cpp 

OBJS += \
./src/Main.o \
./src/Net.o \
./src/Neuron.o \
./src/TrainingData.o 

CPP_DEPS += \
./src/Main.d \
./src/Net.d \
./src/Neuron.d \
./src/TrainingData.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -std=c++11 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



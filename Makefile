# Compiler
CXX = g++

# Compiler Flags
CXXFLAGS = -std=c++17 -I./eigen -I. -Wall -Wextra

# Output Binary Name
TARGET = myserver

# Source Files
SRCS = main.cpp

# Build Rule
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

# Clean Rule
clean:
	rm -f $(TARGET)

mrun:
	make
	./$(TARGET)

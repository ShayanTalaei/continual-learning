#!/bin/bash

# Storage Benchmark Script
# Tests write speeds to different storage locations

echo "=== Storage Write Speed Benchmark ==="
echo "Testing different storage locations..."
echo

# Test parameters
TEST_SIZE="100M"  # 100MB test file
TEST_FILE_LOCAL="/tmp/test_write_speed"
TEST_FILE_HOME="/sailhome/stalaei/test_write_speed"
TEST_FILE_MATX="/matx/u/stalaei/test_write_speed"
TEST_FILE_SCR="/scr/stalaei/test_write_speed"

# Function to test write speed
test_write_speed() {
    local location=$1
    local test_file=$2
    local description=$3
    
    echo "Testing: $description"
    echo "Location: $location"
    
    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$test_file")"
    
    # Test write speed using dd
    echo "Writing $TEST_SIZE to $location..."
    start_time=$(date +%s.%N)
    
    dd if=/dev/zero of="$test_file" bs=1M count=100 2>/dev/null
    exit_code=$?
    
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc)
    
    if [ $exit_code -eq 0 ]; then
        # Calculate speed in MB/s
        speed=$(echo "scale=2; 100 / $duration" | bc)
        echo "✓ Success: ${speed} MB/s"
        echo "  Duration: ${duration} seconds"
    else
        echo "✗ Failed to write to $location"
    fi
    
    # Clean up test file
    rm -f "$test_file"
    echo
}

# Test local storage (/tmp)
test_write_speed "/tmp" "$TEST_FILE_LOCAL" "Local Storage (/tmp)"

# Test home directory
test_write_speed "/sailhome/stalaei" "$TEST_FILE_HOME" "Home Directory (/sailhome/stalaei)"

# Test matx network storage
test_write_speed "/matx/u/stalaei" "$TEST_FILE_MATX" "Network Storage (/matx/u/stalaei)"

# Test scr local storage
test_write_speed "/scr/stalaei" "$TEST_FILE_SCR" "Local Storage (/scr/stalaei)"

# Additional test: Test with smaller chunks to simulate model download
echo "=== Chunked Write Test (simulating model download) ==="
echo "Testing with smaller chunks to simulate downloading model files..."
echo

test_chunked_write() {
    local location=$1
    local test_file=$2
    local description=$3
    
    echo "Testing: $description (chunked writes)"
    echo "Location: $location"
    
    mkdir -p "$(dirname "$test_file")"
    
    start_time=$(date +%s.%N)
    
    # Write in chunks of 10MB (simulating model file downloads)
    for i in {1..10}; do
        dd if=/dev/zero of="${test_file}_chunk_${i}" bs=1M count=10 2>/dev/null
    done
    
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc)
    
    if [ $? -eq 0 ]; then
        speed=$(echo "scale=2; 100 / $duration" | bc)
        echo "✓ Success: ${speed} MB/s"
        echo "  Duration: ${duration} seconds"
    else
        echo "✗ Failed chunked write to $location"
    fi
    
    # Clean up test files
    rm -f "${test_file}_chunk_"*
    echo
}

test_chunked_write "/tmp" "$TEST_FILE_LOCAL" "Local Storage (/tmp)"
test_chunked_write "/sailhome/stalaei" "$TEST_FILE_HOME" "Home Directory (/sailhome/stalaei)"
test_chunked_write "/matx/u/stalaei" "$TEST_FILE_MATX" "Network Storage (/matx/u/stalaei)"
test_chunked_write "/scr/stalaei" "$TEST_FILE_SCR" "Local Storage (/scr/stalaei)"

echo "=== Benchmark Complete ==="
echo "Summary:"
echo "- Local storage (/tmp) should be fastest"
echo "- Network storage (/matx/u/stalaei) will be slowest due to network overhead"
echo "- Use this data to decide where to download models temporarily"

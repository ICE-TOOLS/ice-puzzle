# Ice Puzzle: Bitcoin Private Key Finder (Puzzle #86)

This project is a high-performance, multi-threaded Bitcoin private key finder designed to solve Puzzle #86. It leverages GPU acceleration (CUDA) for rapid elliptic curve cryptography (ECC) and hashing operations, aiming for maximum throughput in scanning large private key ranges.

## Features

*   **Multi-threaded CPU Orchestration**: Efficiently divides the private key search space among multiple CPU threads.
*   **Full GPU Acceleration (CUDA)**: Both private key generation (ECC) and Hash160 computation are performed entirely on the GPU for optimal speed.
*   **Optimized ECC**: Utilizes custom Jacobian coordinate arithmetic for secp256k1 point addition and scalar multiplication on the GPU.
*   **Batch Processing**: Processes large batches of private keys on the GPU to minimize host-device transfer overhead.
*   **Real-time Feedback**: Displays scanning speed (keys/sec) in the console title and logs each checked key with colorized output.
*   **Winner Data Saving**: Upon finding a matching private key, it saves the full private key (hex) and Bitcoin address to `winner.txt`.
*   **Configurable Range**: Supports custom start and end private key ranges.

## Prerequisites

To build and run this project, you will need:

1.  **CMake**: A cross-platform build system.
2.  **Vcpkg**: A C++ package manager for Windows.
    *   Install `OpenSSL`, `GMP`, and `secp256k1` via Vcpkg:
        ```bash
        vcpkg install openssl:x64-windows gmp:x64-windows secp256k1:x64-windows
        ```
3.  **NVIDIA CUDA Toolkit**: Required for GPU acceleration. Ensure it's installed and compatible with your Visual Studio version.
    *   Download from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads).
    *   Verify `nvcc` is in your system's PATH or note its location.
4.  **Visual Studio 2022 (or compatible C++ compiler)**: For compiling the C++ and CUDA code.

## Building the Project

1.  **Clean previous builds (optional but recommended):**
    ```bash
    rmdir /s /q build
    ```

2.  **Configure CMake with Vcpkg and CUDA:**
    Navigate to the project root directory (`c:\Users\<YOURNAME>\OneDrive\Desktop\puzzle`).
    ```bash
    cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
    ```
    *   **Important**: If CMake reports "CUDA Toolkit not found", ensure the CUDA Toolkit is installed correctly and `nvcc` is accessible. You might need to explicitly specify the CUDA compiler path:
        ```bash
        cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe"
        ```
        (Adjust `v13.0` to your installed CUDA version).

3.  **Build the project:**
    ```bash
    cmake --build build --config Release -j
    ```
    The `-j` flag enables parallel compilation, speeding up the build process.

## How to Run

After a successful build, the executable `puzzle86.exe` will be located in `c:\Users\<YOURNAME>\OneDrive\Desktop\puzzle\build\Release\`.

1.  **Navigate to the Release directory:**
    ```bash
    cd build\Release
    ```

2.  **Execute the program:**
    ```bash
    .\puzzle86.exe
    ```

### Configuration

The target Bitcoin address and the private key search range are currently hardcoded in `src/main.cpp`:

*   `const std::string targetAddress = "1K3x5L6G57Y494fDqBfrojD28UJv4s5JcK";`
*   `const char* startHex = "2000000000000000000000";`
*   `const char* endHex = "3fffffffffffffffffffff";`

You can modify these values directly in `src/main.cpp` and rebuild the project to search for different targets or ranges.

## Output

The program provides real-time feedback:

*   **Console Title**: Updates with the current scanning speed in "keys/sec".
*   **Per-Key Log**: For each private key checked, a colored line is printed to the console:
    ```
    KEY <full_private_key_hex> PUB <shortened_pubkey_hex> HASH160 <shortened_hash160_hex> ADDR <shortened_address>
    ```
    The private key is always shown in full, while other fields are shortened for readability.
*   **Winner Notification**: If a matching private key is found, a green "FOUND" message will appear:
    ```
    FOUND <full_private_key_hex> <full_bitcoin_address>
    ```
*   **`winner.txt`**: A file named `winner.txt` will be created in the executable's directory, containing the full private key (hex) and the corresponding Bitcoin address.

## GPU Utilization and Performance

This project is heavily optimized for GPU usage. If you observe low GPU utilization (e.g., below 50%), consider the following:

*   **Ensure CUDA is active**: Verify that the project built with `HAVE_CUDA=1` (check CMake output).
*   **Increase Batch Size**: The `batch` constant in `src/main.cpp` (currently `65536`) can be increased further if your GPU has sufficient memory. Larger batches keep the GPU busy longer.
*   **Reduce Console Output**: Printing every checked key to the console can introduce CPU-side bottlenecks. For maximum raw speed, you might consider commenting out the per-key `std::cout` line in `src/main.cpp` and relying solely on the speed in the title and the `winner.txt` file.
*   **Advanced Optimizations (Future)**:
    *   **Multi-stream Pipelining**: Overlapping host-to-device (H2D) and device-to-host (D2H) memory transfers with kernel execution.
    *   **Device-side Sequential Addition**: Implementing `+G` operations directly on the GPU to avoid per-thread scalar multiplication, which is more efficient for sequential key scanning.
    *   **Batched Affine Conversion**: Amortizing the cost of elliptic curve point inversions over many points.

Feel free to experiment with these settings to achieve the highest possible performance on your hardware.

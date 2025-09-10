## 🔧 Build from Source and Run Tests

1. **Install Rust**
   Visit [https://rustup.rs/](https://rustup.rs/) and follow the instructions to install Rust and Cargo.

2. **Install Dependencies**
   Make sure the following packages are installed:

   * `libssl-dev` (OpenSSL)
   * `libz3-dev`
   * On Linux, you will likely also need **Clang ≥ 12.0.0**. Set it as your C compiler using:

     ```bash
     export CC=clang
     export CXX=clang++
     ```

3. **Build the Project**
   Use Cargo to build the project. This may take several seconds to several minutes depending on your system:

   ```bash
   cargo build --release
   ```

4. **Prepare Smart Contract Binaries**
   Each smart contract should be **compiled** before testing. Make sure each contract folder contains the corresponding `.abi` and `.bin` files.

5. **Run Fuzzing on a Contract**
   To test a single smart contract, run:

   ```bash
   cargo run --features print_txn_corpus --package ityfuzz --bin ityfuzz -- evm -t dir_path/*
   ```

   Replace `dir_path` with the actual path to the contract directory.

6. **Log Output**
   For each smart contract tested, output will be saved to a log file named:

   ```
   <contract_name>.log
   ```

   This file is saved in the corresponding contract’s directory. The log contains:

   * **STDOUT**: Basic execution information such as deployment, mutation actions, and coverage results.
   * **STDERR**: Any error messages or internal diagnostics.
   * **Detected Vulnerabilities Section**:
     If any vulnerabilities are found, they will be clearly listed, including:

     * Vulnerability type (e.g., Selfdestruct, Reentrancy)
     * Related function calls and traces
     * Final instruction and branch coverage

     You can quickly locate this section by searching for:

     ```
     Found vulnerabilities!
     ```
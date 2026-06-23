"""Constants for GPU bitshuffle+LZ4 compression."""

# Bitshuffle block size (must match hdf5plugin format)
BLOCK_SIZE: int = 8192

# LZ4 compression constants
LZ4_HASH_LOG: int = 12
LZ4_HASH_SIZE: int = 1 << LZ4_HASH_LOG  # 4096
LZ4_WARP_SIZE: int = 32

# Compression output overhead per block
LZ4_MAX_OVERHEAD: int = 16 + BLOCK_SIZE // 255

# CUDA kernel launch parameters
CUDA_MAX_GRID_DIM_Z: int = 65535
THREADS_PER_BLOCK_COMPACT: int = 256
THREADS_PER_BLOCK_LZ4: int = 32

# Async pipeline batch count (empirically optimal for ~64K frames)
PIPELINE_BATCH_COUNT: int = 32

# Bitshuffle header sizes
BITSHUFFLE_HEADER_SIZE: int = 12  # 8 bytes uncompressed size + 4 bytes block size
BLOCK_HEADER_SIZE: int = 4  # 4 bytes compressed size per block

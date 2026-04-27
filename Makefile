# Define the root directories to search in
SRC_DIR := src
CPP_DIR := src/pointnet2
ROOT_DIR := .
ROOTDIR = $(shell pwd)
# Define a variable for fast execution
UV_FAST = uv run --no-sync
.PHONY: sync clean

clean-build:
	@echo "Removing build artifacts and compiled extensions..."
	# Remove standard build directories
	find $(ROOT_DIR) -path $(ROOT_DIR)/.venv -prune -o -type d -name "build" -exec rm -rf {} + 2>/dev/null
	find $(ROOT_DIR) -path $(ROOT_DIR)/.venv -prune -o -type d -name "dist" -exec rm -rf {} + 2>/dev/null
	find $(ROOT_DIR) -path $(ROOT_DIR)/.venv -prune -o -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null
# 	# Remove compiled C++ shared objects (.so files) only from src, not venv
# 	find $(SRC_DIR) -type f -name "*.so" -delete
	# Remove scikit-build specific artifacts if any (but not from venv)
	find $(ROOT_DIR) -path $(ROOT_DIR)/.venv -prune -o -type d -name "_skbuild" -exec rm -rf {} + 2>/dev/null
	# Remove editable install artifacts from venv  
	@if [ -d ".venv" ]; then \
		rm -f .venv/lib/python3.11/site-packages/__editable__*.pth 2>/dev/null; \
		rm -f .venv/lib/python3.11/site-packages/__editable__*finder.py 2>/dev/null; \
		rm -rf .venv/lib/python3.11/site-packages/pointnet2_cuda-*.dist-info 2>/dev/null; \
		echo "Cleaned C++ build artifacts and package metadata"; \
	fi

clear-cache:
	@echo "Clearing UV cache..."
	uv cache clear
	@echo "UV cache cleared."

sync:
	@echo "Cleaning build artifacts..."
	$(MAKE) clean-build
	@echo "Locking dependencies with recent pyproject.toml..."
	uv lock
	@echo "Building with uv sync (this will install Python dependencies)..."
	uv sync
	@echo "Building C++ extensions..."
	$(MAKE) build-cpp
	@echo "Cleaning build artifacts..."
	$(MAKE) clean-build
	@echo "Sync complete!"

build-cpp:
	@echo "Building C++ extensions..."
	@( cd $(CPP_DIR) && PYTHON=$(ROOTDIR)/.venv/bin/python && $$PYTHON setup.py build_ext --inplace )
	@echo "All C++ extensions built!"

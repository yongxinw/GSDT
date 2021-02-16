CUDA=$1
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.4.0+${CUDA}.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.4.0+${CUDA}.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.4.0+${CUDA}.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.4.0+${CUDA}.html
pip install torch-geometric
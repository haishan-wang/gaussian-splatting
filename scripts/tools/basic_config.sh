#!/bin/env bash
# conda enviroment activation
echo Set basic config... Current server: [$SERVER_NAME]
if [[ $SERVER_NAME == 'desktop' ]]; then
    module load miniconda
    export PYTHONPATH=/l/wangh18/Projects/GSCOMP/test-compgs
else
    module load mamba  
    module load gcc/10.5.0
fi
source activate compgs

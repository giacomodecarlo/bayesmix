#!/usr/bin/env bash

#  build/run_mcmc_mfa \
#  --algo-params-file examples/mfa_hierarchy/in/algo.asciipb \
#  --hier-type MFA --hier-args examples/mfa_hierarchy/in/mfa_auto.asciipb \
#  --mix-type DP --mix-args examples/mfa_hierarchy/in/dp_gamma.asciipb \
#  --coll-name examples/mfa_hierarchy/out/chains_auto.recordio \
#  --data-file examples/mfa_hierarchy/in/data.csv \
#  --grid-file examples/mfa_hierarchy/in/data.csv \
#  --dens-file examples/mfa_hierarchy/out/density_file_auto.csv \
#  --n-cl-file examples/mfa_hierarchy/out/numclust_auto.csv \
#  --clus-file examples/mfa_hierarchy/out/clustering_auto.csv \
#  --best-clus-file resources/tutorial/out/best_clustering.csv

./build/run_mcmc_mfa examples/mfa_hierarchy/config.txt

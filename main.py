# To run:
# python3 -B main.py

from toy_analysis import *

# Different ways of building vectors are being explored, e.g.
# dw = DeepWalk
# mb = Membership-based
embedding_method = "mb"

### Main ###
load_all_atoms()
# load_deepwalk_model()

# populate_atomspace()
# generate_subsets()
# calculate_truth_values()
# infer_attractions()
# export_all_atoms()

calculate_fuzzy_membership_values()

if embedding_method == "dw":
  train_deepwalk_model()
  export_deepwalk_model()
elif embedding_method == "mb":
  build_property_vectors()

compare(embedding_method)
# plot_pca(embedding_method)

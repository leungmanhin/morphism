from analysis import *

# Different ways of building vectors are being explored, e.g.
# dw = DeepWalk
# mb = Membership-based
embedding_method = "mb"

### Main ###
# load_all_atomes()
# load_deepwalk_model()

populate_atomspace()
generate_subsets()
calculate_truth_values()
infer_attractions()
export_all_atoms()
# train_deepwalk_model()
# export_deepwalk_model()
# plot_pca(deepwalk[deepwalk.wv.vocab], deepwalk.wv.vocab)
# compare(deepwalk)

build_property_vectors()
plot_pca(list(property_vector_dict.values()), list(property_vector_dict.keys()))
compare(property_vector_dict)
# print_property_prevalence()

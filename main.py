from analysis import *

### Main ###
# load_all_atomes()
# load_deepwalk_model()

populate_atomspace()
generate_subsets()
calculate_truth_values()
infer_attractions()
export_all_atoms()
build_property_vectors()
train_deepwalk_model()
export_deepwalk_model()
plot_pca()
compare()

print_property_prevalence()

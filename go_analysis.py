import numpy
import os
import pickle
import random
from gensim.models import Word2Vec
from matplotlib import pyplot
from opencog.atomspace import AtomSpace, types
from opencog.bindlink import execute_atom
from opencog.logger import log
from opencog.scheme_wrapper import scheme_eval
from opencog.type_constructors import *
from opencog.utilities import initialize_opencog
from scipy.spatial import distance
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.decomposition import PCA, KernelPCA

log.set_level("ERROR")

base_datasets_dir = os.getcwd() + "/datasets/"
base_results_dir = os.getcwd() + "/results/go/"
if not os.path.exists(base_results_dir):
  os.makedirs(base_results_dir)

go_scm = base_datasets_dir + "GO_2020-04-01.scm"
go_annotation_scm = base_datasets_dir + "GO_annotation_gene-level_2020-04-01.scm"
member_links_scm = base_results_dir + "member-links.scm"
inheritance_links_scm = base_results_dir + "inheritance-links.scm"
subset_links_scm = base_results_dir + "subset-links.scm"
attraction_links_scm = base_results_dir + "attraction-links.scm"
sentences_pickle = base_results_dir + "sentences.pickle"
deepwalk_bin = base_results_dir + "deepwalk.bin"
results_csv = base_results_dir + "results.csv"

go_term_prefix = "GO:"

property_vectors = {}
deepwalk = None
num_sentences = 10000000
num_walks = 9

### Initialize the AtomSpace ###
atomspace = AtomSpace()
initialize_opencog(atomspace)

### Guile setup ###
def scm(atomese):
  return scheme_eval(atomspace, atomese).decode("utf-8")

scm("(add-to-load-path \"/usr/share/guile/site/2.2/opencog\")")
scm("(add-to-load-path \".\")")
scm("(use-modules (opencog) (opencog bioscience) (opencog ure) (opencog pln))")
scm(" ".join([
  "(define (write-atoms-to-file file atoms)",
    "(define fp (open-output-file file))",
    "(for-each",
      "(lambda (x) (display x fp))",
      "atoms)",
    "(close-port fp))"]))

def build_property_vectors():
  print("--- Building property vectors")
  global property_vectors

  for go_term in get_concepts(go_term_prefix):
    p_vec = []
    for subset in atomspace.get_atoms_by_type(types.SubsetLink):
      tv = subset.tv
      # Only look at the ones loaded directly from data, i.e. having (stv 1 1),
      # and excluded the inferred (inversed) ones
      if tv.mean == 1 and tv.confidence == 1 and subset.out[0] == go_term:
        pat = subset.out[1]
        attraction = AttractionLink(go_term, pat)
        attraction_tv = attraction.tv
        p_vec.append(attraction.tv.mean * attraction.tv.confidence)
    property_vectors[go_term.name] = p_vec

def calculate_truth_values():
  print("--- Calculating Truth Values")

  node_member_dict = {}
  def get_members(node):
    if node_member_dict.get(node):
      return node_member_dict[node]
    else:
      members = [x.out[0] for x in node.incoming if x.type == types.MemberLink and x.out[1] == node]
      node_member_dict[node] = members
      return members

  def get_confidence(count):
    return float(scm("(count->confidence " + str(count) + ")"))

  # MemberLinks are generated directly from the data, can be considered as true
  for m in atomspace.get_atoms_by_type(types.MemberLink):
    m.tv = TruthValue(1, 1)

  # ConceptNode "A" (stv s c)
  # where:
  # s = |A| / |universe|
  # c = |universe|
  # TODO: Register this Atom type in Python
  GeneNode = 221
  universe_size = len(atomspace.get_atoms_by_type(GeneNode))
  tv_confidence = get_confidence(universe_size)
  for c in get_concepts(go_term_prefix):
    member_size = len(get_members(c))
    tv_strength = member_size / universe_size
    c.tv = TruthValue(tv_strength, tv_confidence)

  # SubLinks are generated directly from the data, and is true by definition
  for s in atomspace.get_atoms_by_type(types.SubsetLink):
    s.tv = TruthValue(1, 1)

  # Infer the inverse subsets
  scm(" ".join([
    "(define (true-subset-inverse S)",
      "(let* ((A (gar S))",
             "(B (gdr S))",
             "(ATV (cog-tv A))",
             "(BTV (cog-tv B))",
             "(A-positive-count (* (cog-tv-mean ATV) (cog-tv-count ATV)))",
             "(B-positive-count (* (cog-tv-mean BTV) (cog-tv-count BTV)))",
             "(TV-strength (if (< 0 B-positive-count)",
                              "(exact->inexact (/ A-positive-count B-positive-count))",
                              "1))",
             "(TV-count B-positive-count)",
             "(TV-confidence (count->confidence TV-count))",
             "(TV (stv TV-strength TV-confidence)))",
        "(Subset TV B A)))"]))
  scm("(map true-subset-inverse (cog-get-atoms 'SubsetLink))")

def compare(embedding_method):
  print("--- Comparing PLN vs " + embedding_method)

  def get_go_terms_in_hierarchy():
    go_terms = set()
    for subset in atomspace.get_atoms_by_type(types.SubsetLink):
      tv = subset.tv
      # Only look at the ones loaded directly from data, i.e. having (stv 1 1),
      # and excluded the inferred (inversed) ones
      if tv.mean == 1 and tv.confidence == 1:
        go_terms.add(subset.out[0].name)
        go_terms.add(subset.out[1].name)
    return list(go_terms)

  node_pattern_dict = {}
  def get_properties(node):
    def get_attractions(node):
      return [x for x in node.incoming if x.type == types.AttractionLink and x.out[0] == node]
    if node_pattern_dict.get(node):
      return node_pattern_dict[node]
    else:
      # Filter out the ones with mean == 0
      attractions = [x for x in get_attractions(node) if x.tv.mean > 0]
      pats = [x.out[1] for x in attractions]
      node_pattern_dict[node] = pats
    return pats

  # Get the user pairs
  print("--- Generating GO term pairs")
  gos = get_go_terms_in_hierarchy()
  random.shuffle(gos)
  go_pairs = list(zip(gos[::2], gos[1::2]))

  print("--- Generating results")
  # PLN setup
  scm("(pln-load 'empty)")
  scm("(pln-add-rule \"intensional-similarity-direct-introduction-rule\")")

  # Generate the results
  all_rows = []
  for pair in go_pairs:
    go1 = pair[0]
    go2 = pair[1]
    go1_properties = get_properties(ConceptNode(go1))
    go2_properties = get_properties(ConceptNode(go2))
    go1_pattern_size = len(go1_properties)
    go2_pattern_size = len(go2_properties)
    common_properties = set(go1_properties).intersection(go2_properties)
    common_pattern_size = len(common_properties)
    # PLN intensional similarity
    intsim_tv = intensional_similarity(go1, go2)
    intsim_tv_mean = intsim_tv.mean if intsim_tv.confidence > 0 else 0
    # Vector distance
    v1 = deepwalk[go1]
    v2 = deepwalk[go2]
    # vec_dist = distance.euclidean(v1, v2)
    vec_dist = distance.cosine(v1, v2)
    # vec_dist = fuzzy_jaccard(v1, v2)
    # vec_dist = tanimoto(v1, v2)
    row = [
      go1,
      go2,
      go1_pattern_size,
      go2_pattern_size,
      common_pattern_size,
      intsim_tv_mean,
      vec_dist]
    all_rows.append(row)

  # Sort in descending order of intensional similarity
  intsim_col_idx = 5
  vecdist_col_idx = 6
  all_rows = numpy.array(all_rows)
  all_rows_sorted = all_rows[all_rows[:, intsim_col_idx].argsort()][::-1]

  intsim_columns = all_rows_sorted[:, intsim_col_idx].astype(numpy.float)
  vecdist_columns = all_rows_sorted[:, vecdist_col_idx].astype(numpy.float)
  pearson = pearsonr(intsim_columns, vecdist_columns)[0]
  spearman = spearmanr(intsim_columns, vecdist_columns)[0]
  kendall = kendalltau(intsim_columns, vecdist_columns)[0]

  print("Pearson = {}\nSpearman = {}\nKendall = {}".format(pearson, spearman, kendall))

  # Write to file
  with open(results_csv, "w") as f:
    first_row = ",".join([
      "GO 1",
      "GO 2",
      "No. of GO 1 properties",
      "No. of GO 2 properties",
      "No. of common properties",
      "Intensional Similarity",
      "Vector Distance"])
    f.write(first_row + "\n")
    for row in all_rows_sorted:
      f.write(",".join(row) + "\n")

def do_kpca():
  def kernel_func(X):
    dist_dict = {}
    mat = []
    i = 0
    cnt = 0
    total = X.shape[0] ** 2
    for a in X:
      row = []
      j = 0
      for b in X:
        cnt += 1
        print("--- Working on {}/{}...".format(cnt, total))
        if (i, j) in dist_dict:
          row.append(dist_dict[(i, j)])
        elif (j, i) in dist_dict:
          row.append(dist_dict[(j, i)])
        else:
          # dist = fuzzy_jaccard(a, b)
          dist = tanimoto(a, b)
          row.append(dist)
          dist_dict[(i, j)] = dist
        j += 1
      mat.append(row)
      i += 1
    return numpy.array(mat)

  print("--- Doing KPCA")

  X = numpy.array(list(property_vectors.values()))
  X_kpca = KernelPCA(kernel = "precomputed").fit_transform(kernel_func(X))

  for k, kpca_v in zip(property_vectors.keys(), X_kpca):
    property_vectors[k] = kpca_v

def do_pca():
  print("--- Doing PCA")
  pca = PCA()
  pca_results = pca.fit_transform(list(property_vectors.values()))
  cum_sum = numpy.cumsum(pca.explained_variance_ratio_)
  if cum_sum[-1] >= 1:
    print("--- Best PCA components is {} (sum to 1)".format(numpy.where(cum_sum >= 1)[0][0] + 1))
  else:
    print("--- Best PCA components is {} (sum to {})".format(pca.n_components_, cum_sum[-1]))
  for k, pca_v in zip(property_vectors.keys(), pca_results):
    property_vectors[k] = pca_v

def export_all_atoms():
  def write_atoms_to_file(filename, atom_list_str):
    if not os.path.exists(filename):
      open(filename, "w").close()
    scm(" ".join([
      "(write-atoms-to-file",
      "\"" + filename + "\"",
      atom_list_str + ")"]))

  print("--- Exporting Atoms to files")
  write_atoms_to_file(member_links_scm, "(cog-get-atoms 'MemberLink)")
  write_atoms_to_file(inheritance_links_scm, "(cog-get-atoms 'InheritanceLink)")
  write_atoms_to_file(subset_links_scm, "(cog-get-atoms 'SubsetLink)")
  write_atoms_to_file(attraction_links_scm, "(cog-get-atoms 'AttractionLink)")

def export_deepwalk_model():
  print("--- Exporting DeepWalk model to \"{}\"".format(deepwalk_bin))
  deepwalk.save(deepwalk_bin)

def export_property_vectors():
  print("--- Exporting property vectors to \"{}\"".format(property_vector_pickle))
  with open(property_vector_pickle, "wb") as f:
    pickle.dump(property_vectors, f)

def fuzzy_jaccard(v1, v2):
  '''
  This function is supposed to reflect the actual calculation done in the PLN rule.
  Check out the 'intensional-similarity-direct-introduction-rule' for details.
  '''
  numerator = 0
  denominator = 0
  for x, y in zip(v1, v2):
    # The "intersect"
    if x > 0 and y > 0:
      numerator = numerator + min(x,y)
    # The "union"
    denominator = denominator + max(x,y)
  tvs = (numerator / denominator) if denominator > 0 else 0
  return tvs

def get_concepts(str_prefix):
  return [x for x in atomspace.get_atoms_by_type(types.ConceptNode) if x.name.startswith(str_prefix)]

def infer_attractions():
  print("--- Inferring AttractionLinks")
  scm("(pln-load 'empty)")
  # (Subset A B) |- (Subset (Not A) B)
  scm("(pln-add-rule \"subset-condition-negation-rule\")")
  # (Subset A B) (Subset (Not A) B) |- (Attraction A B)
  scm("(pln-add-rule \"subset-attraction-introduction-rule\")")
  scm(" ".join(["(pln-bc",
                  "(Attraction (Variable \"$X\") (Variable \"$Y\"))",
                  "#:vardecl",
                    "(VariableSet",
                      "(TypedVariable (Variable \"$X\") (Type \"ConceptNode\"))",
                      "(TypedVariable (Variable \"$Y\") (Type \"ConceptNode\")))",
                  "#:maximum-iterations 12",
                  "#:complexity-penalty 10)"]))

def generate_subsets():
  print("--- Inferring subsets and new members")
  scm("(pln-load 'empty)")
  scm("(pln-load-from-path \"rules/translation.scm\")")
  scm("(pln-load-from-path \"rules/transitivity.scm\")")
  # (Inheritance C1 C2) |- (Subset C1 C2)
  scm("(pln-add-rule \"present-inheritance-to-subset-translation-rule\")")
  # (Subset C1 C2) (Subset C2 C3) |- (Subset C1 C3)
  scm("(pln-add-rule \"present-subset-transitivity-rule\")")
  # (Member G C1) (Subset C1 C2) |- (Member G C2)
  scm("(pln-add-rule \"present-mixed-member-subset-transitivity-rule\")")
  scm(" ".join([
    "(pln-fc",
      "(Inheritance (Variable \"$X\") (Variable \"$Y\"))",
      "#:vardecl",
        "(VariableSet",
          "(TypedVariable (Variable \"$X\") (Type \"ConceptNode\"))",
          "(TypedVariable (Variable \"$Y\") (Type \"ConceptNode\")))",
      "#:maximum-iterations 12",
      "#:fc-full-rule-application #t)"]))

def intensional_similarity(c1, c2):
  cn1 = "(Concept \"{}\")".format(c1)
  cn2 = "(Concept \"{}\")".format(c2)
  intsim = "(IntensionalSimilarity {} {})".format(cn1, cn2)
  scm("(pln-bc {})".format(intsim))
  tv_mean = float(scm("(cog-mean {})".format(intsim)))
  tv_conf = float(scm("(cog-confidence {})".format(intsim)))
  return TruthValue(tv_mean, tv_conf)

def load_all_atoms():
  print("--- Loading Atoms from files")
  scm("(use-modules (opencog persist-file))")
  scm("(load-file \"" + member_links_scm + "\")")
  scm("(load-file \"" + inheritance_links_scm + "\")")
  scm("(load-file \"" + subset_links_scm + "\")")
  scm("(load-file \"" + attraction_links_scm + "\")")

def load_deepwalk_model():
  global deepwalk
  print("--- Loading an existing model from \"{}\"".format(deepwalk_bin))
  deepwalk = Word2Vec.load(deepwalk_bin)

def load_property_vectors():
  global property_vectors
  print("--- Loading property vectors from \"{}\"".format(property_vector_pickle))
  property_vectors = pickle.load(open(property_vector_pickle, "rb"))

def populate_atomspace():
  print("--- Populating the AtomSpace")
  scm("(use-modules (opencog persist-file))")
  scm("(load-file \"" + go_scm + "\")")
  scm("(load-file \"" + go_annotation_scm + "\")")

def tanimoto(v1, v2):
  v1_v2 = numpy.dot(v1, v2)
  v1_sq = numpy.sum(numpy.square(v1))
  v2_sq = numpy.sum(numpy.square(v2))
  return v1_v2 / (v1_sq + v2_sq - v1_v2)

def train_deepwalk_model():
  global deepwalk
  go_terms_involved = set()
  next_words_dict = {}
  sentences = []

  def add_to_next_words_dict(w, nw):
    if next_words_dict.get(w):
      next_words_dict[w].add(nw)
    else:
      next_words_dict[w] = {nw}

  print("--- Gathering next words")
  for subset in atomspace.get_atoms_by_type(types.SubsetLink):
    tv = subset.tv
    # Only look at the ones loaded directly from data, i.e. having (stv 1 1),
    # and excluded the inferred (inversed) ones
    # Getting from SubsetLinks instead of InheritanceLinks, so that the one
    # inferred via transitivity will be included in the sequence
    if tv.mean == 1 and tv.confidence == 1:
      child = subset.out[0].name
      parent = subset.out[1].name
      pred = "inherits-geneontologyterm"
      rev_pred = "geneontologyterm-inherited-by"
      add_to_next_words_dict(child, (pred, parent))
      add_to_next_words_dict(parent, (rev_pred, child))
      go_terms_involved.add(child)
      go_terms_involved.add(parent)

  for k, v in next_words_dict.items():
    next_words_dict[k] = list(v)

  print("--- Generating sentences")
  first_words = tuple(go_terms_involved)
  for i in range(num_sentences):
    sentence = [random.choice(first_words)]
    for j in range(num_walks):
      last_word = sentence[-1]
      next_words = random.choice(next_words_dict.get(last_word))
      sentence.append(next_words[0])
      sentence.append(next_words[1])
    sentences.append(sentence)
    if len(sentences) % 10000 == 0:
      print(len(sentences))
  with open(sentences_pickle, "wb") as f:
    pickle.dump(sentences, f)

  print("--- Training model")
  deepwalk = Word2Vec(sentences, min_count=1)

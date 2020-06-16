import os
import pickle
import random
from gensim.models import Word2Vec
from matplotlib import pyplot
from goatools import obo_parser
from goatools.semantic import deepest_common_ancestor
from opencog.atomspace import AtomSpace, types
from opencog.logger import log
from opencog.scheme_wrapper import scheme_eval
from opencog.type_constructors import *
from opencog.utilities import initialize_opencog
from scipy.spatial import distance
from sklearn.decomposition import PCA

log.set_level("ERROR")

# Whether to consider subsets to be true by definition and assign (stv 1 1) to them
subset_true_by_definition = True

go_basic_obo = os.getcwd() + "/datasets/go-basic.obo"
go_scm = os.getcwd() + "/datasets/GO_2020-04-01.scm"
go_annotation_scm = os.getcwd() + "/datasets/GO_annotation_gene-level_2020-04-01.scm"
member_links_scm = os.getcwd() + "/results/member-links.scm"
inheritance_links_scm = os.getcwd() + "/results/inheritance-links.scm"
subset_links_scm = os.getcwd() + "/results/subset-links.scm"
attraction_links_scm = os.getcwd() + "/results/attraction-links.scm"
sentences_pickle = os.getcwd() + "/results/sentences.pickle"
deepwalk_bin = os.getcwd() + "/results/deepwalk.bin"
pca_png = os.getcwd() + "/results/pca.png"
results_csv = os.getcwd() + "/results/results.csv"

go_term_prefix = "GO:"

deepwalk = None
go_dag = obo_parser.GODag(go_basic_obo)

### Utils ###
def scm(atomese):
  return scheme_eval(atomspace, atomese).decode("utf-8")

def get_concepts(name_prefix):
  return list(
           filter(
             lambda x : x.name.startswith(name_prefix),
             atomspace.get_atoms_by_type(types.ConceptNode)))

def intensional_difference(c1, c2):
  cn1 = "(Concept \"{}\")".format(c1)
  cn2 = "(Concept \"{}\")".format(c2)
  intdiff = "(IntensionalDifference {} {})".format(cn1, cn2)
  scm("(pln-bc {})".format(intdiff))
  tv_mean = float(scm("(cog-mean {})".format(intdiff)))
  tv_conf = float(scm("(cog-confidence {})".format(intdiff)))
  return TruthValue(tv_mean, tv_conf)

def intensional_similarity(c1, c2):
  cn1 = "(Concept \"{}\")".format(c1)
  cn2 = "(Concept \"{}\")".format(c2)
  intsim = "(IntensionalSimilarity {} {})".format(cn1, cn2)
  scm("(pln-bc {})".format(intsim))
  tv_mean = float(scm("(cog-mean {})".format(intsim)))
  tv_conf = float(scm("(cog-confidence {})".format(intsim)))
  return TruthValue(tv_mean, tv_conf)

### Initialize the AtomSpace ###
atomspace = AtomSpace()
initialize_opencog(atomspace)

### Guile setup ###
scm("(add-to-load-path \"/usr/share/guile/site/2.2/opencog\")")
scm("(add-to-load-path \".\")")
scm("(use-modules (opencog) (opencog bioscience) (opencog ure) (opencog pln) (opencog persist-file))")
scm(" ".join([
  "(define (write-atoms-to-file file atoms)",
    "(define fp (open-output-file file))",
    "(for-each",
      "(lambda (x) (display x fp))",
      "atoms)",
    "(close-port fp))"]))

def load_all_atomes():
  print("--- Loading Atoms from files")
  scm("(load-file \"" + member_links_scm + "\")")
  scm("(load-file \"" + inheritance_links_scm + "\")")
  scm("(load-file \"" + subset_links_scm + "\")")
  scm("(load-file \"" + attraction_links_scm + "\")")

def export_all_atoms():
  def write_atoms_to_file(filename, atom_list_str):
    scm(" ".join([
      "(write-atoms-to-file",
      "\"" + filename + "\"",
      atom_list_str + ")"]))

  print("--- Exporting Atoms to files")
  write_atoms_to_file(member_links_scm, "(cog-get-atoms 'MemberLink)")
  write_atoms_to_file(inheritance_links_scm, "(cog-get-atoms 'InheritanceLink)")
  write_atoms_to_file(subset_links_scm, "(cog-get-atoms 'SubsetLink)")
  write_atoms_to_file(attraction_links_scm, "(cog-get-atoms 'AttractionLink)")

def load_deepwalk_model():
  global deepwalk
  print("--- Loading an existing model from \"{}\"".format(deepwalk_bin))
  deepwalk = Word2Vec.load(deepwalk_bin)

def export_deepwalk_model():
  global deepwalk
  deepwalk.save(deepwalk_bin)

### Populate the AtomSpace ###
def populate_atomspace():
  print("--- Populating the AtomSpace")
  scm("(load-file \"" + go_scm + "\")")
  scm("(load-file \"" + go_annotation_scm + "\")")

### Infer subsets and members ###
def infer_subsets_and_members():
  print("--- Inferring subsets and new members")
  scm("(pln-load 'empty)")
  scm("(pln-load-from-path \"rules/translation.scm\")")
  scm("(pln-load-from-path \"rules/transitivity.scm\")")
  # (Inheritance C1 C2) |- (Subset C1 C2)
  scm("(pln-add-rule-by-name \"present-inheritance-to-subset-translation-rule\")")
  # (Subset C1 C2) (Subset C2 C3) |- (Subset C1 C3)
  scm("(pln-add-rule-by-name \"present-subset-transitivity-rule\")")
  # (Member G C1) (Subset C1 C2) |- (Member G C2)
  scm("(pln-add-rule-by-name \"present-mixed-member-subset-transitivity-rule\")")
  scm(" ".join([
    "(pln-fc",
      "(Inheritance (Variable \"$X\") (Variable \"$Y\"))",
      "#:vardecl",
        "(VariableSet",
          "(TypedVariable (Variable \"$X\") (Type \"ConceptNode\"))",
          "(TypedVariable (Variable \"$Y\") (Type \"ConceptNode\")))",
      "#:maximum-iterations 12",
      "#:fc-full-rule-application #t)"]))
  if not subset_true_by_definition:
    for s in atomspace.get_atoms_by_type(types.SubsetLink):
      node1 = s.out[0]
      node2 = s.out[1]
      SubsetLink(node2, node1)

def calculate_truth_values():
  print("--- Calculating Truth Values")

  node_member_dict = {}
  def get_members(node):
    if node_member_dict.get(node):
      return node_member_dict[node]
    else:
      memblinks = filter(lambda x : x.type == types.MemberLink and x.out[1] == node, node.incoming)
      members = [x.out[0] for x in tuple(memblinks)]
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
  universe_size = int(scm("(length (cog-get-atoms 'GeneNode))"))
  tv_confidence = get_confidence(universe_size)
  for c in get_concepts(go_term_prefix):
    member_size = len(get_members(c))
    tv_strength = member_size / universe_size
    c.tv = TruthValue(tv_strength, tv_confidence)

  if subset_true_by_definition:
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
  else:
    # SubsetLink (stv s c)
    #   A
    #   B
    # where:
    # s = |B intersect A| / |A|
    # c = |A|
    for s in atomspace.get_atoms_by_type(types.SubsetLink):
      node1 = s.out[0]
      node2 = s.out[1]
      node1_members = get_members(node1)
      node2_members = get_members(node2)
      common_members = set(node2_members).intersection(node1_members)
      tv_strength = len(common_members) / len(node1_members) if len(node1_members) > 0 else 0
      tv_confidence = get_confidence(len(node1_members))
      s.tv = TruthValue(tv_strength, tv_confidence)

def infer_attractions():
  print("--- Inferring AttractionLinks")
  scm("(pln-load 'empty)")
  # (Subset A B) |- (Subset (Not A) B)
  scm("(pln-add-rule-by-name \"subset-condition-negation-rule\")")
  # (Subset A B) (Subset (Not A) B) |- (Attraction A B)
  scm("(pln-add-rule-by-name \"subset-attraction-introduction-rule\")")
  scm(" ".join(["(pln-bc",
                  "(Attraction (Variable \"$X\") (Variable \"$Y\"))",
                  "#:vardecl",
                    "(VariableSet",
                      "(TypedVariable (Variable \"$X\") (Type \"ConceptNode\"))",
                      "(TypedVariable (Variable \"$Y\") (Type \"ConceptNode\")))",
                  "#:maximum-iterations 12",
                  "#:complexity-penalty 10)"]))

def train_deepwalk_model():
  global deepwalk
  next_words_dict = {}
  sentences = []

  def add_to_next_words_dict(w, nw):
    if next_words_dict.get(w):
      next_words_dict[w].add(nw)
    else:
      next_words_dict[w] = {nw}

  print("--- Gathering next words")
  inhlinks = atomspace.get_atoms_by_type(types.InheritanceLink)
  for inhlink in inhlinks:
    child = inhlink.out[0].name
    parent = inhlink.out[1].name
    pred = "inherits-geneontologyterm"
    rev_pred = "geneontologyterm-inherited-by"
    add_to_next_words_dict(child, (pred, parent))
    add_to_next_words_dict(parent, (rev_pred, child))

  memblinks = atomspace.get_atoms_by_type(types.MemberLink)
  for memblink in memblinks:
    child = memblink.out[0].name
    parent = memblink.out[1].name
    pred = "in-gene-ontology"
    rev_pred = "has-gene-ontology-member"
    add_to_next_words_dict(child, (pred, parent))
    add_to_next_words_dict(parent, (rev_pred, child))

  for k, v in next_words_dict.items():
    next_words_dict[k] = list(v)

  print("--- Generating sentences")
  num_sentences = 10000000
  num_walks = 7
  first_words = [x.name for x in get_concepts(go_term_prefix)]
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

def plot_pca():
  print("--- Plotting")
  X = deepwalk[deepwalk.wv.vocab]
  pca = PCA(n_components = 2)
  result = pca.fit_transform(X)
  pyplot.scatter(result[:, 0], result[:, 1])
  words = list(deepwalk.wv.vocab)
  for i, word in enumerate(words):
    pyplot.annotate(word, xy = (result[i, 0], result[i, 1]))
  pyplot.savefig(pca_png, dpi=1000)

def compare():
  print("--- Comparing PLN vs DW")

  node_pattern_dict = {}
  def get_properties(node):
    def get_attractions(node):
      return list(
               filter(
                 lambda x : x.type == types.AttractionLink and x.out[0] == node,
                 node.incoming))
    if node_pattern_dict.get(node):
      return node_pattern_dict[node]
    else:
      attractions = list(filter(lambda x : x.tv.mean > 0, get_attractions(node)))
      pats = [x.out[1] for x in attractions]
      node_pattern_dict[node] = pats
    return pats

  def dag_distance(go1, go2):
    try:
      common_go = deepest_common_ancestor([go1, go2], go_dag)
      go1_depth = go_dag[go1].depth
      go2_depth = go_dag[go2].depth
      common_go_depth = go_dag[common_go].depth
      return go1_depth + go2_depth - 2 * common_go_depth
    except:
      return "NA"

  # Get the user pairs
  print("--- Generating GO term pairs")
  gos = [x.name for x in get_concepts(go_term_prefix)]
  random.shuffle(gos)
  go_pairs = list(zip(gos[::2], gos[1::2]))

  print("--- Generating results")
  # PLN setup
  scm("(pln-load 'empty)")
  # TODO: Check the size of the universe
  scm("(pln-add-rule-by-name \"intensional-difference-direct-introduction-rule\")")
  scm("(pln-add-rule-by-name \"intensional-similarity-direct-introduction-rule\")")

  # Output file
  results_csv_fp = open(results_csv, "w")
  first_row = ",".join([
    "GO 1",
    "GO 2",
    "GO 1 depth",
    "GO 2 depth",
    "Distance in DAG",
    "No. of GO 1 properties",
    "No. of GO 2 properties",
    "No. of common properties",
    "Intensional Difference (GO1 GO2)",
    "Intensional Difference (GO2 GO1)",
    "Intensional Similarity",
    "Vector distance"])
  results_csv_fp.write(first_row + "\n")

  # Generate the results
  for pair in go_pairs:
    go1 = pair[0]
    go2 = pair[1]
    go1_depth = go_dag[go1].depth if go_dag.get(go1) else "NA"
    go2_depth = go_dag[go2].depth if go_dag.get(go2) else "NA"
    dag_dist = dag_distance(go1, go2)
    go1_properties = get_properties(ConceptNode(go1))
    go2_properties = get_properties(ConceptNode(go2))
    go1_pattern_size = len(go1_properties)
    go2_pattern_size = len(go2_properties)
    common_properties = set(go1_properties).intersection(go2_properties)
    common_pattern_size = len(common_properties)
    # PLN intensional difference
    intdiff_go1_go2_tv = intensional_difference(go1, go2)
    intdiff_go1_go2_tv_mean = intdiff_go1_go2_tv.mean if intdiff_go1_go2_tv.confidence > 0 else 0
    intdiff_go2_go1_tv = intensional_difference(go2, go1)
    intdiff_go2_go1_tv_mean = intdiff_go2_go1_tv.mean if intdiff_go2_go1_tv.confidence > 0 else 0
    intsim_tv = intensional_similarity(go1, go2)
    intsim_tv_mean = intsim_tv.mean if intsim_tv.confidence > 0 else 0
    # DeepWalk euclidean distance
    v1 = deepwalk[go1]
    v2 = deepwalk[go2]
    vec_dist = distance.euclidean(v1, v2)
    row = ",".join([
      go1,
      go2,
      str(go1_depth),
      str(go2_depth),
      str(dag_dist),
      str(go1_pattern_size),
      str(go2_pattern_size),
      str(common_pattern_size),
      str(intdiff_go1_go2_tv_mean),
      str(intdiff_go2_go1_tv_mean),
      str(intsim_tv_mean),
      str(vec_dist)])
    results_csv_fp.write(row + "\n")
  results_csv_fp.close()

### Main ###
# load_all_atomes()
# load_deepwalk_model()

populate_atomspace()
infer_subsets_and_members()
calculate_truth_values()
infer_attractions()
export_all_atoms()
# XXX: Workaround the issue of getting SubsetLinks as well when only InheritanceLinks are expected
[atomspace.remove(s) for s in atomspace.get_atoms_by_type(types.SubsetLink)]
train_deepwalk_model()
export_deepwalk_model()
# plot_pca()

compare()

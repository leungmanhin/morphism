import os
import pickle
import random
from gensim.models import Word2Vec
from matplotlib import pyplot
from opencog.atomspace import AtomSpace, types
from opencog.scheme_wrapper import scheme_eval
from opencog.type_constructors import *
from opencog.utilities import initialize_opencog
from scipy.spatial import distance
from sklearn.decomposition import PCA

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

deepwalk = None

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
  for c in get_concepts("GO:"):
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

### Main ###
# load_all_atomes()
# load_deepwalk_model()

populate_atomspace()
infer_subsets_and_members()
calculate_truth_values()
# infer_attractions()
# export_all_atoms()
# train_deepwalk_model()
# export_deepwalk_model()
# plot_pca()

# compare()

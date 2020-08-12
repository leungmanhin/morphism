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
from sklearn.decomposition import PCA, FastICA

log.set_level("ERROR")

base_results_dir = os.getcwd() + "/results/toy/"

member_links_scm = base_results_dir + "member-links.scm"
evaluation_links_scm = base_results_dir + "evaluation-links.scm"
subset_links_scm = base_results_dir + "subset-links.scm"
attraction_links_scm = base_results_dir + "attraction-links.scm"
sentences_pickle = base_results_dir + "sentences.pickle"
deepwalk_bin = base_results_dir + "deepwalk.bin"
pca_png = base_results_dir + "pca.png"
results_csv = base_results_dir + "results.csv"

instance_prefix = "instance:"
person_prefix = "person:"
property_prefix = "property:"

deepwalk = None
property_vector_dict = {}

num_people = 100
num_properties = 1000
num_properties_per_person = 10
fixed_num_properties_per_person = False
num_sentences = 10000000
num_walks = 18
pca_components = 100
ica_components = 47

### Utils ###
def scm(atomese):
  return scheme_eval(atomspace, atomese).decode("utf-8")

def get_concepts(str_prefix):
  return [x for x in atomspace.get_atoms_by_type(types.ConceptNode) if x.name.startswith(str_prefix)]

def get_people_with_property(prop):
  return execute_atom(atomspace,
           GetLink(
             EvaluationLink(
               PredicateNode("has_property"),
               ListLink(
                 VariableNode("$X"),
                 prop)))).out

def get_concept_properties(concept):
  return execute_atom(atomspace,
           GetLink(
             EvaluationLink(
               PredicateNode("has_property"),
               ListLink(
                 concept,
                 VariableNode("$X"))))).out

def print_property_prevalence():
  property_cnt = 0
  sum_prevalence = 0
  for p in get_concepts(property_prefix):
    prevalence = len(get_people_with_property(p))
    property_cnt = property_cnt + 1
    sum_prevalence = sum_prevalence + prevalence
    print("{}, {}".format(p.name, prevalence))
  print("Total no. of people: {}\nTotal no. of properties: {}\nAverage property prevalence: {} ({}%)".format(
      len(get_concepts(person_prefix)),
      property_cnt,
      sum_prevalence/property_cnt,
      100 * sum_prevalence/property_cnt/num_people))

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
scm("(use-modules (opencog) (opencog ure) (opencog pln))")
scm(" ".join([
  "(define (write-atoms-to-file file atoms)",
    "(define fp (open-output-file file))",
    "(for-each",
      "(lambda (x) (display x fp))",
      "atoms)",
    "(close-port fp))"]))

def load_all_atoms():
  print("--- Loading Atoms from files")
  scm("(use-modules (opencog persist-file))")
  scm("(load-file \"" + member_links_scm + "\")")
  scm("(load-file \"" + evaluation_links_scm + "\")")
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
  write_atoms_to_file(evaluation_links_scm, "(cog-get-atoms 'EvaluationLink)")
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

  # Create people and the properties linked to them
  for i in range(0, num_people):
    instance_conceptnode = ConceptNode(instance_prefix + str(i))
    person_conceptnode = ConceptNode(person_prefix + str(i))

    MemberLink(
      instance_conceptnode,
      person_conceptnode)

    if fixed_num_properties_per_person:
      num_person_properties = num_properties_per_person
    else:
      num_person_properties = random.randint(1, num_properties_per_person)
    print("No. of properties person-{} has = {}".format(i, num_person_properties))

    for j in random.sample(range(num_properties), num_person_properties):
      property_conceptnode = ConceptNode(property_prefix + str(j))

      MemberLink(
        instance_conceptnode,
        property_conceptnode)

      EvaluationLink(
        PredicateNode("has_property"),
        ListLink(
          person_conceptnode,
          property_conceptnode))

def generate_subsets():
  print("--- Generating SubsetLinks")
  for evalink in atomspace.get_atoms_by_type(types.EvaluationLink):
    source = evalink.out[1].out[0]
    target = evalink.out[1].out[1]
    SubsetLink(source, target)
    SubsetLink(target, source)

def calculate_truth_values():
  print("--- Calculating Truth Values")

  node_member_dict = {}
  def get_members(node):
    if node_member_dict.get(node):
      return node_member_dict[node]
    else:
      memblinks = [x for x in node.incoming if x.type == types.MemberLink and x.out[1] == node]
      members = [x.out[0] for x in tuple(memblinks)]
      node_member_dict[node] = members
      return members

  def get_confidence(count):
    return float(scm("(count->confidence " + str(count) + ")"))

  # EvaluationLinks are generated directly from the data, can be considered as true
  for e in atomspace.get_atoms_by_type(types.EvaluationLink):
    e.tv = TruthValue(1, 1)

  # MemberLinks are generated directly from the data, can be considered as true
  for m in atomspace.get_atoms_by_type(types.MemberLink):
    m.tv = TruthValue(1, 1)

  # ConceptNode "A" (stv s c)
  # where:
  # s = |A| / |universe|
  # c = |universe|
  universe_size = len(get_concepts(instance_prefix))
  tv_confidence = get_confidence(universe_size)
  for c in atomspace.get_atoms_by_type(types.ConceptNode):
    member_size = len(get_members(c))
    tv_strength = member_size / universe_size
    c.tv = TruthValue(tv_strength, tv_confidence)

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
    tv_strength = len(common_members) / len(node1_members)
    tv_confidence = get_confidence(len(node1_members))
    s.tv = TruthValue(tv_strength, tv_confidence)

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

def train_deepwalk_model():
  global deepwalk
  next_words_dict = {}
  sentences = []

  def add_to_next_words_dict(w, nw):
    if next_words_dict.get(w):
      next_words_dict[w].add(nw)
    else:
      next_words_dict[w] = {nw}

  def get_reverse_pred(pred):
    if pred == "has_property":
      return "is_a_property_of"

  print("--- Gathering next words")
  evalinks = atomspace.get_atoms_by_type(types.EvaluationLink)
  for evalink in evalinks:
    pred = evalink.out[0].name
    rev_pred = get_reverse_pred(pred)
    source_node = evalink.out[1].out[0]
    target_node = evalink.out[1].out[1]
    if source_node.type == types.VariableNode or target_node.type == types.VariableNode:
      continue
    source = source_node.name
    target = target_node.name
    add_to_next_words_dict(source, (pred, target))
    add_to_next_words_dict(target, (rev_pred, source))
  for k, v in next_words_dict.items():
    next_words_dict[k] = tuple(v)

  print("--- Generating sentences")
  first_words = [x.name for x in get_concepts(person_prefix)]
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

def build_property_vectors(pca = False, ica = False):
  global property_vectors
  all_properties = get_concepts(property_prefix)

  for person in get_concepts(person_prefix):
    pvec = []
    for p in all_properties:
      attraction = AttractionLink(person, p)
      attraction_tv = attraction.tv
      pvec.append(attraction.tv.mean * attraction.tv.confidence)
    property_vector_dict[person.name] = pvec
  # Do PCA for the sparse vectors
  if pca:
    pca = PCA(n_components = pca_components)
    pca_results = pca.fit_transform(list(property_vector_dict.values()))
    for k, pca_v in zip(property_vector_dict.keys(), pca_results):
      property_vector_dict[k] = pca_v
  # Do ICA for the sparse vectors
  if ica:
    ica = FastICA(n_components = ica_components)
    ica_results = ica.fit_transform(list(property_vector_dict.values()))
    for k, ica_v in zip(property_vector_dict.keys(), ica_results):
      property_vector_dict[k] = ica_v

def plot_pca(embedding_method):
  print("--- Plotting")
  if embedding_method == "DW":
    X = deepwalk[deepwalk.wv.vocab]
    labels = deepwalk.wv.vocab
  elif embedding_method == "FMBPV":
    X = list(property_vector_dict.values())
    labels = list(property_vector_dict.keys())
  pca = PCA(n_components = 2)
  result = pca.fit_transform(X)
  pyplot.scatter(result[:, 0], result[:, 1])
  words = list(labels)
  for i, word in enumerate(words):
    pyplot.annotate(word, xy = (result[i, 0], result[i, 1]))
  pyplot.savefig(pca_png, dpi=1000)

def compare(embedding_method):
  print("--- Comparing PLN vs " + embedding_method)

  # Get the pairs
  print("--- Generating pairs")
  people = [x.name for x in get_concepts(person_prefix)]
  # For getting random pairs
  # random.shuffle(people)
  # people_pairs = list(zip(people[::2], people[1::2]))
  people_pairs = []
  for person1 in people:
    for person2 in people:
      if person1 != person2 and {person1, person2} not in people_pairs:
        people_pairs.append({person1, person2})
  people_pairs = [tuple(x) for x in people_pairs]

  print("--- Generating results")

  # PLN setup
  scm("(pln-load 'empty)")
  scm("(pln-add-rule \"intensional-similarity-direct-introduction-rule\")")

  # Generate the results
  all_rows = []
  for pair in people_pairs:
    p1 = pair[0]
    p2 = pair[1]
    p1_properties = get_concept_properties(ConceptNode(p1))
    p2_properties = get_concept_properties(ConceptNode(p2))
    p1_pattern_size = len(p1_properties)
    p2_pattern_size = len(p2_properties)
    common_properties = set(p1_properties).intersection(p2_properties)
    common_pattern_size = len(common_properties)
    # PLN intensional similarity
    intsim = intensional_similarity(p1, p2)
    intsim_tv = intsim.mean if intsim.confidence > 0 else 0
    # Vector distance
    if embedding_method == "DW":
      v1 = deepwalk[p1]
      v2 = deepwalk[p2]
    elif embedding_method == "FMBPV":
      v1 = property_vector_dict[p1]
      v2 = property_vector_dict[p2]
    # vec_dist = distance.euclidean(v1, v2)
    # vec_dist = distance.cityblock(v1, v2)
    vec_dist = distance.jaccard(v1, v2)
    # vec_dist = fuzzy_jaccard(p1, p2, v1, v2)
    row = [
      p1,
      p2,
      p1_pattern_size,
      p2_pattern_size,
      common_pattern_size,
      intsim_tv,
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
      "Person 1",
      "Person 2",
      "No. of person 1 properties",
      "No. of person 2 properties",
      "No. of common properties",
      "Intensional Similarity",
      "Vector Distance"])
    f.write(first_row + "\n")
    for row in all_rows_sorted:
      f.write(",".join(row) + "\n")

def fuzzy_jaccard(p1, p2, v1, v2):
  numerator = 0
  denominator = 0
  for x, y in zip(v1, v2):
    # The "intersect"
    if x > 0 and y > 0:
      numerator = numerator + min(x,y)
    denominator = denominator + max(x,y)
  tvs = (numerator / denominator) if denominator > 0 else 0
  return tvs

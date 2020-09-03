# Overview
The goal here is to find out if there is a correspondence between the nodes in the OpenCog intensional reasoning and their corresponding vectors built using various methods. Motivation and preliminary results are detailed in:
- Paper: https://arxiv.org/abs/2005.12535
- Video: https://underline.io/lecture/778-embedding-vector-differences-can-be-aligned-with-uncertain-intensional-logic-differences-

For making the embedding vectors, two methods have been implemented at the moment:

1) DeepWalk

Makes use of the Word2Vec language model to make embeddings for the nodes (as opposed to words in linguistic context) in the AtomSpace. The "sentences" are built by taking "walks" from one node to another through their interconnected links. The likelihood of a walk being selected is based on the TruthValue of each of the links connected to a particular node being looked at during "sentence" construction (as opposed to selecting randomly as done in the original DeepWalk algorithm). On the other hand, the datasets we have been using here are fairly simple, the properties associated with the concepts are "crisp", so the walk are basically chosen randomly. But when a different dataset is used, additional logic should be implemented to handle probabilistic walk selecting.

2) Fuzzy-membership based

Embedding of a node is built, with an entry correspond to a property found in the AtomSpace, and the value being the TruthValue of the AttractionLink, which reflects if a particular property is more likely to be a property of that node than other nodes.

After building the vectors, some dimension reduction technique can optionally be done (but is actually recommended for the vectors built using the 2nd method because those are very sparse vectors in most cases). Two techniques are implemented at the moment -- PCA and KPCA. For KPCA, two kernel functions can be used:
1) Tanimoto -- Compute the Tanimoto distance between two vectors
2) Fuzzy Jaccard -- Compute the intensional similarity using the actual calculation used in the PLN intensional similarity direct introduction rule

The steps in an experiment involved:
1) Converting the data into Atomese
2) Generate `SubsetLinks` and `AttractionLinks` for the concepts that we are interested in, and calculate TruthValues for them (for PLN intensional reasoning)
3) Create embedding vectors for the same set of concepts, using either one of the available embedding methods
4) Randomly select pairs of concepts, and for each pair, compute the intensional similarity and vector distance between them
5) Calculate the correlation between the intensional similarities and vector distances calculated in the previous step

# To Run
In this repository, there are two Python scripts, correspond to either the toy dataset that's made for testing purpose only, or the [Social Network: MOOC User Action Dataset](https://snap.stanford.edu/data/act-mooc.html).

There is an additional script `main.py`, which is used to control what functions to be called for the experiment, e.g. load embeddings vs generate embeddings, be sure to check out what's in there, comment/uncomment out the parts that are needed or not, before running it via the following command:

```
python3 -B main.py
```

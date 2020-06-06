import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import time

from decision_tree import *

class Random_Forest:

  # =========================================================

  def __init__(self, max_depth, trimmed_feature_size, sample_size, forest_size):

    # Initializing data
    self.X_train = None
    self.y_train = None

    # Decision Tree Params
    self.max_depth = max_depth
    self.trimmed_feature_size = trimmed_feature_size
    self.sample_size = sample_size

    # Forest Params
    self.trees = []
    self.tree_structures = []
    self.forest_size = forest_size

  # =========================================================

  # Section 1: Forest Fitting Functions
  def fit(self, X_train, y_train):
    self.X_train = X_train
    self.y_train = y_train
    self.build_forest()

  # =========================================================

  def build_forest(self):

    # For each tree, by index.
    for tree_num in range(0, self.forest_size):

      # Create a unique bagged data set lunch :)
      data = self.return_bagged_data_for_one_tree()

      # Initialize the tree and pass it a depth limit
      tree = Decision_Tree(self.max_depth)

      # Fit the tree to the unique data
      tree.fit(data)

      # Take the dictionary which represents the tree's structure
      tree_structure = tree.get_tree_structure()

      # Append the Decision_Tree object to unique array.
      self.trees.append(tree)

      # Append Decision_Tree object structure dictionary to unique array.
      self.tree_structures.append(tree_structure)

  # =========================================================

  # Prepare a trimmed set of data for a single tree's usage.
  def return_bagged_data_for_one_tree(self):
    trimmed_feature_set = self.propose_trimmed_feature_set()
    tree_specific_bagged_data = self.one_bag_data(trimmed_feature_set)
    return tree_specific_bagged_data

  # Take a set of columns, return a set of data corresponding only to those columns.
  def one_bag_data(self, trimmed_feature_set):
    bagged_merged_data = self.X_train.copy()
    bagged_merged_data['label'] = self.y_train
    bagged_merged_data = bagged_merged_data[trimmed_feature_set].sample(self.sample_size)
    return bagged_merged_data

  # Propose a set of features to be used for each tree.
  def propose_trimmed_feature_set(self):
    random_feature_indices = random.sample(
                                          range(0, len(self.X_train.columns)),
                                          self.trimmed_feature_size)
    sample_columns = self.X_train.columns[random_feature_indices]
    sample_columns = sample_columns.append(pd.Index(['label']))
    return sample_columns

  # =========================================================

  # Section 2: Forest Prediction Functions

  # =========================================================

  def predict(self, X_set):
    tree_votes = []
    for tree in self.trees:
      tree_votes.append(tree.predict(X_set))
    consolidated_votes = self.consolidate_votes(tree_votes)
    return self.final_consolidation(consolidated_votes)

  # =========================================================

  # Take the index of the class with the highest vote number.
  def final_consolidation(self, tally_card):
    output_arr = []
    for item in tally_card:
      output_arr.append(item.index(max(item)))
    return output_arr

  # =========================================================

  # Sum all the individual votes into appropriate entry in tally_card.
  def consolidate_votes(self, votes):
    num_votes_per_tree = len(votes[0])
    num_trees = len(votes)
    tally_card = self.create_tally_card(num_votes_per_tree, num_trees)
    for i in range(0, num_trees):
      for j in range(0, num_votes_per_tree):
        tally_card[j][0] += votes[i][j][0]
        tally_card[j][1] += votes[i][j][1]
    return tally_card

  # =========================================================

  def create_tally_card(self, num_votes_per_tree, num_trees):
    tally_card = []
    for i in range(0, num_votes_per_tree):
      tally_card.append([0, 0])
    return tally_card

  # =========================================================

  # Section 3: Utility Functions
  def return_forest(self):
    return self.trees

  # =========================================================

  def return_tree_structures(self):
    return self.tree_structures

  # =========================================================

  def load_forest(self, list_of_structures):
    self.tree_structures = list_of_structures
    for structure in self.tree_structures:
      tree = Decision_Tree(self.max_depth)
      tree.load_structure(structure)
      self.trees.append(tree)

  # =========================================================

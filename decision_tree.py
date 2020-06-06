import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import time

class Decision_Tree:

  # =========================================================

  def __init__(self, max_depth):
    self.max_depth = max_depth
    self.structure = {0: {'data' : None, 'class_ratios' : [None, None],
                          'split_col' : None, 'split_val' : None,
                          'impurity' : None}}
    self.depth_counter = 1
    self.node_counter = 1

  # =========================================================

  # Section 1: Fit Functions
  def fit(self, data):

    self.structure.update({0 : {'data' : data, 'class_ratios' : self.get_class_ratios(data),
                                'split_col' : None, 'split_val' : None,
                                'impurity' : None}})
    self.build_tree()
    self.clean_trees()

  # =========================================================

  def build_tree(self):
    # For each depth level
    for depth in range(0, self.max_depth):
      # For each node at the current depth
      for node_index in self.get_current_level_indices():
        # If the node exists and was not skipped.
        if (node_index in self.structure.keys()):
          # If the node is not a leaf.
          if self.structure[node_index]['impurity'] != 0:
            # Acquire the best split with the lowest impurity
            best_impurity, best_split_col, best_split_val = self.get_best_split(self.structure[node_index]['data'])
            # If Impurity is 0, nothing can be gained from a split. Set the current node to a leaf.
            if best_impurity  < 0.05:
              self.structure[node_index].update({'impurity' : 0})
              self.node_counter += 2
              # print('No children for you')
              # print(self.node_counter)
            # Otherwise, splitting is worthwhile.
            else:
              # Update current node with the best split
              self.structure[node_index].update({'split_col' : best_split_col, 'split_val' : best_split_val, 'impurity' : best_impurity})

              # Divide the children's data along the split.
              # print(self.structure[node_index]['data'])
              # print(self.structure[node_index]['data'][best_split_col])
              left_child_data = self.structure[node_index]['data'].loc[self.structure[node_index]['data'][best_split_col] <= best_split_val]
              right_child_data = self.structure[node_index]['data'].loc[self.structure[node_index]['data'][best_split_col] > best_split_val]
              # print(self.node_counter)

              # Do these updates if the new level should be leaves.
              if depth == (self.max_depth - 1):
                self.structure.update({self.node_counter : {'data' : left_child_data, 'class_ratios' : self.get_class_ratios(left_child_data),
                                                            'split_col' : None, 'split_val' : None,
                                                            'impurity' : 0}})
                self.node_counter += 1
                # print(self.node_counter)
                self.structure.update({self.node_counter : {'data' : right_child_data, 'class_ratios' : self.get_class_ratios(right_child_data),
                                                            'split_col' : None, 'split_val' : None,
                                                            'impurity' : 0}})
                self.node_counter += 1
              # Do these if the next level is beginning or intermediate.
              else:
                self.structure.update({self.node_counter : {'data' : left_child_data, 'class_ratios' : self.get_class_ratios(left_child_data),
                                                            'split_col' : None, 'split_val' : None,
                                                            'impurity' : None}})
                self.node_counter += 1
                # print(self.node_counter)
                self.structure.update({self.node_counter : {'data' : right_child_data, 'class_ratios' : self.get_class_ratios(right_child_data),
                                                            'split_col' : None, 'split_val' : None,
                                                            'impurity' : None}})
                self.node_counter += 1
          # If the node is a leaf, skip, we'll evaluate the children of the next node in the queue.
          else:
            self.node_counter += 2
            # print('No children for you')
            # print(self.node_counter)
            pass
        # if the node does not exist, it was skipped because its parent had an impurity of 0. Pass.
        else:
          pass
      # Increment depth.
      self.depth_counter += 1

  # =========================================================

  def get_best_split(self, data):
    best_impurity = 999
    best_split_col = None
    best_split_val = None
    candidate_splits = self.get_candidate_splits(data)
    for candidate_col in candidate_splits.keys():
      impurity = self.evaluate_split(data, candidate_col, candidate_splits[candidate_col][0])
      if impurity < best_impurity:
        best_impurity = impurity
        best_split_col = candidate_col
        best_split_val = candidate_splits[candidate_col][0]
    return best_impurity, best_split_col, best_split_val

  # =========================================================

  def get_candidate_splits(self, data):
    output_dict = {}
    for col in data.columns:
      if col != 'label':
        val_array = data[col].unique()
        if len(val_array) != 0:
          val_min = data[col].unique().min()
          val_max = data[col].unique().max()
          if len(val_array) > 2:
            output_dict.update({col : [np.random.uniform(val_min, val_max, 1)[0]]})
          else:
            output_dict.update({col : [val_min]})
        else:
          pass
    return output_dict

  # =========================================================

  def evaluate_split(self, data, s_col, s_val):
    Sl = data.loc[data[s_col] <= s_val]['label']
    Sr = data.loc[data[s_col] > s_val]['label']
    impurity = self.gini(Sl, Sr)
    return impurity

  # =========================================================

  def return_proportions(self, pd_series, size):
    count = 0
    for i in range(size):
      if pd_series.iloc[i] == 0:
        count += 1
    return (count / size), 1 - (count / size)

  # =========================================================

  def gini(self, Sl, Sr):
    size_Sl, size_Sr = Sl.shape[0], Sr.shape[0]
    # size_Sr = Sr.shape[0]
    if size_Sl > 0:
      p0l, p1l = self.return_proportions(Sl, size_Sl)
    else:
      p0l, p1l = 0, 0
    if size_Sr > 0:
      p0r, p1r = self.return_proportions(Sr, size_Sr)
    else:
      p0r, p1r = 0, 0
    prop_Sl = size_Sl / (size_Sl + size_Sr)
    prop_Sr = size_Sr / (size_Sl + size_Sr)
    return ((1 - (p0l**2 + p1l**2)) * prop_Sl) + ((1 - (p0r**2 + p1r**2)) * prop_Sr)

  # =========================================================

  def get_class_ratios(self, data):
    total = data.shape[0] + 1
    zeroes = data.loc[data['label'] == 0].shape[0]
    ones = data.loc[data['label'] == 1].shape[0]
    return [zeroes/total, ones/total]

  # =========================================================

  # ----------------------------------#
  # Section 2: Prediction Functions   #
  # ----------------------------------#

  def predict(self, X_set):
    # Where each vote in votes is this tree's class ratio for a specific example.
    votes = []
    for X_num in range(0, X_set.shape[0]):
      # print(X_set)
      votes.append(self.secure_them_votes_breh(X_set.iloc[X_num]))
    # print(votes)
    return votes

  # =========================================================

  def secure_them_votes_breh(self, X_row):
    current_node = 0
    while self.structure[current_node]['impurity'] != 0:
      if X_row[self.structure[current_node]['split_col']] <= self.structure[current_node]['split_val']:
        current_node_candidate = self.get_left_child_node_index(current_node)
        if current_node_candidate not in self.structure.keys():
          break
        else:
          current_node = current_node_candidate
      else:
        current_node_candidate = self.get_right_child_node_index(current_node)
        if current_node_candidate not in self.structure.keys():
          break
        else:
          current_node = current_node_candidate
    return self.structure[current_node]['class_ratios']

  # =========================================================

  # Section 3: Utility Functions

  # =========================================================

  def get_current_level_indices(self):
    index_arr = []
    for i in range(2**(self.depth_counter - 1) - 1, 2**(self.depth_counter) - 1):
      index_arr.append(i)
    return index_arr

  # =========================================================

  def get_tree_structure(self):
    return self.structure

  # =========================================================

  def load_structure(self, structure):
    self.structure = structure

  # =========================================================

  def get_parent_node_index(self, n):
    return np.floor((n - 1) / 2)

  # =========================================================

  def get_left_child_node_index(self, n):
    return (2*n) + 1

  # =========================================================

  def get_right_child_node_index(self, n):
    return (2*n) + 2

  # =========================================================

  def print_node_structure(self):
    for key in self.structure.keys():
      print('-----------------------------------')
      print(key)
      print(self.structure[key]['class_ratios'])
      print(self.structure[key]['split_col'])
      print(self.structure[key]['split_val'])
      print(self.structure[key]['impurity'])

  # =========================================================

  def clean_trees(self):
    for key in self.structure.keys():
      self.structure[key].update({'data' : None})

  # =========================================================

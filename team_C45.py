import math
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import chi2_contingency

class _DecisionNode:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}

    def depth(self):
        if len(self.children) == 0:
            return 1
        else:
            max_depth = 0
            for child in self.children.values():
                if isinstance(child, _DecisionNode):
                    child_depth = child.depth()
                    if child_depth > max_depth:
                        max_depth = child_depth
            return max_depth + 1

    def add_child(self, value, node):
        self.children[value] = node

    def count_leaves(self):
        if len(self.children) == 0:
            return 1
        else:
            count = 0
            for child in self.children.values():
                if isinstance(child, _DecisionNode):
                    count += child.count_leaves()
                else:
                    count += 1
            return count

    def count_nodes(self):
        count = 1  # Count the current node
        for child in self.children.values():
            if isinstance(child, _DecisionNode):
                count += child.count_nodes()
            else:
                count += 1
        return count

class _LeafNode:
    def __init__(self, label, weight):
        self.label = label
        self.weight = weight

class C45Classifier:
    def __init__(self, max_depth=None, min_samples_split=2, chi_square_threshold=0.05):
        self.tree = None
        self.attributes = None
        self.data = None
        self.weight = 1
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.chi_square_threshold = chi_square_threshold

    def __calculate_entropy(self, data, weights):
        class_counts = {}
        total_weight = 0.0

        for i, record in enumerate(data):
            label = record[-1]
            weight = weights[i]

            if label not in class_counts:
                class_counts[label] = 0.0
            class_counts[label] += weight
            total_weight += weight

        entropy                                 = 0.0

        for count in class_counts.values():
            probability = count / total_weight
            entropy -= probability * math.log2(probability)

        return entropy

    def __split_data(self, data, attribute_index, attribute_value, weights):
        split_data = []
        split_weights = []

        for i, record in enumerate(data):
            if record[attribute_index] == attribute_value:
                split_data.append(record[:attribute_index] + record[attribute_index+1:])
                split_weights.append(weights[i])

        return split_data, split_weights

    def __select_best_attribute_c50(self, data, attributes, weights):
        total_entropy = self.__calculate_entropy(data, weights)
        best_attribute = None
        best_gain_ratio = 0.0
        split_info = 0.0
        for attribute_index in range(len(attributes)):
            attribute_values = set([record[attribute_index] for record in data])
            attribute_entropy = 0.0

            for value in attribute_values:
                subset, subset_weights = self.__split_data(data, attribute_index, value, weights)
                subset_entropy = self.__calculate_entropy(subset, subset_weights)
                subset_probability = sum(subset_weights) / sum(weights)
                attribute_entropy += subset_probability * subset_entropy
                split_info -= subset_probability * math.log2(subset_probability)

            gain = total_entropy - attribute_entropy

            if split_info != 0.0:
                gain_ratio = gain / split_info
            else:
                gain_ratio = 0.0

            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_attribute = attribute_index

        return best_attribute

    def __majority_class(self, data, weights):
        class_counts = {}

        for i, record in enumerate(data):
            label = record[-1]
            weight = weights[i]

            if label not in class_counts:
                class_counts[label] = 0.0
            class_counts[label] += weight

        majority_class = None
        max_count = 0.0

        for label, count in class_counts.items():
            if count > max_count:
                max_count = count
                majority_class = label

        return majority_class

    def __chi_square_test(self, data, attribute_index):
        contingency_table = pd.crosstab([record[attribute_index] for record in data], [record[-1] for record in data])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        return p

    def __build_decision_tree(self, data, attributes, weights, depth=0):
        class_labels = set([record[-1] for record in data])

        if len(class_labels) == 1:
            return _LeafNode(class_labels.pop(), sum(weights))

        if len(attributes) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return _LeafNode(self.__majority_class(data, weights), sum(weights))

        if len(data) < self.min_samples_split:
            return _LeafNode(self.__majority_class(data, weights), sum(weights))

        best_attribute = self.__select_best_attribute_c50(data, attributes, weights)

        if best_attribute is None:
            return _LeafNode(self.__majority_class(data, weights), sum(weights))

        # Kiểm định Chi-square
        p_value = self.__chi_square_test(data, best_attribute)
        if p_value > self.chi_square_threshold:
            return _LeafNode(self.__majority_class(data, weights), sum(weights))

        best_attribute_name = attributes[best_attribute]
        tree = _DecisionNode(best_attribute_name)
        attributes = attributes[:best_attribute] + attributes[best_attribute+1:]
        attribute_values = set([record[best_attribute] for record in data])

        for value in attribute_values:
            subset, subset_weights = self.__split_data(data, best_attribute, value, weights)

            if len(subset) == 0:
                tree.add_child(value, _LeafNode(self.__majority_class(data, weights), sum(subset_weights)))
            else:
                tree.add_child(value, self.__build_decision_tree(subset, attributes, subset_weights, depth + 1))

        return tree

    def __make_tree(self, data, attributes, weights):
        return self.__build_decision_tree(data, attributes, weights)

    def __train(self, data, weight=1):
        self.weight = weight
        self.attributes = data.columns.tolist()[:-1]
        weights = [self.weight] * len(data)
        self.tree = self.__make_tree(data.values.tolist(), self.attributes, weights)
        self.data = data

    def __classify(self, tree=None, instance=[]):
        if self.tree is None:
            raise Exception('Decision tree has not been trained yet!')

        if tree is None:
            tree = self.tree

        if isinstance(tree, _LeafNode):
            return tree.label

        attribute = tree.attribute
        attribute_index = self.attributes.index(attribute)
        attribute_values = instance[attribute_index]

        if attribute_values in tree.children:
            child_node = tree.children[attribute_values]
            return self.__classify(child_node, instance)
        else:
            class_labels = []
            for child_node in tree.children.values():
                if isinstance(child_node, _LeafNode):
                    class_labels.append(child_node.label)
            if len(class_labels) == 0:
                return self.__majority_class(self.data.values.tolist(), [1.0] * len(self.data))
            majority_class = max(set(class_labels))
            return majority_class

    def fit(self, data, label, weight=1):
        if isinstance(data, pd.DataFrame):
            data = pd.concat([data, label], axis=1)
        else:
            data = pd.DataFrame(np.c_[data, label])
        self.__train(data, weight)

    def predict(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.values.tolist()
        elif isinstance(data, list) and isinstance(data[0], dict):
            data = [list(d.values()) for d in data]

        if len(data[0]) != len(self.attributes):
            raise Exception('Number of variables in data and attributes do not match!')

        return [self.__classify(None, record) for record in data]

    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)

        if isinstance(y_test, pd.Series):
            y_test = y_test.values.tolist()

        acc = {}
        true_pred = 0
        real_acc = {}
        for i in range(len(y_test)):
            if y_test[i] not in real_acc:
                real_acc[y_test[i]] = 0
            real_acc[y_test[i]] += 1
            if y_test[i] == y_pred[i]:
                if y_test[i] not in acc:
                    acc[y_test[i]] = 0
                acc[y_test[i]] += 1
                true_pred += 1
        for key in acc:
            acc[key] /= real_acc[key]

        total_acc = true_pred / len(y_test)
        print("Evaluation result: ")
        print("Total accuracy: ", total_acc)
        for key in acc:
            print("Accuracy ", key, ": ", acc[key])

    def prune(self, x_val, y_val, sample_size=1000, n_jobs=-1):
        def prune_node(node, x_val_sample, y_val_sample):
            if isinstance(node, _LeafNode):
                return node

            if isinstance(node, _DecisionNode):
                for value, child in node.children.items():
                    node.children[value] = prune_node(child, x_val_sample, y_val_sample)

                if all(isinstance(child, _LeafNode) for child in node.children.values()):
                    leaf_labels = [child.label for child in node.children.values()]
                    majority_label = max(set(leaf_labels), key=leaf_labels.count)
                    leaf_node = _LeafNode(majority_label, sum(child.weight for child in node.children.values()))

                    original_error = self.__calculate_error(x_val_sample, y_val_sample)
                    self.__replace_node(node, leaf_node)
                    pruned_error = self.__calculate_error(x_val_sample, y_val_sample)

                    if pruned_error <= original_error:
                        return leaf_node
                    else:
                        self.__replace_node(leaf_node, node)

            return node

        # Chuyển đổi các mảng numpy thành DataFrame của pandas
        if isinstance(x_val, np.ndarray):
            x_val = pd.DataFrame(x_val)
        if isinstance(y_val, np.ndarray):
            y_val = pd.Series(y_val)

        # Lấy mẫu ngẫu nhiên nhỏ hơn từ tập dữ liệu kiểm tra
        if sample_size < len(x_val):
            sample_indices = np.random.choice(len(x_val), sample_size, replace=False)
            x_val_sample = x_val.iloc[sample_indices]
            y_val_sample = y_val.iloc[sample_indices]
        else:
            x_val_sample = x_val
            y_val_sample = y_val

        # Sử dụng xử lý song song để cắt tỉa cây
        self.tree = prune_node(self.tree, x_val_sample, y_val_sample)

    def __calculate_error(self, x_val, y_val):
        y_pred = self.predict(x_val)
        errors = sum(1 for true, pred in zip(y_val, y_pred) if true != pred)
        return errors / len(y_val)

    def __calculate_accuracy(self, x_val, y_val):
        y_pred = self.predict(x_val)
        correct_predictions = sum(1 for true, pred in zip(y_val, y_pred) if true == pred)
        return correct_predictions / len(y_val)

    def __replace_node(self, old_node, new_node):
        def replace_recursive(node):
            if node == old_node:
                return new_node
            if isinstance(node, _DecisionNode):
                for value, child in node.children.items():
                    node.children[value] = replace_recursive(child)
            return node

        self.tree = replace_recursive(self.tree)

    def generate_tree_diagram(self, graphviz, filename):
        dot = graphviz.Digraph()

        def build_tree(node, parent_node=None, edge_label=None):
            if isinstance(node, _DecisionNode):
                current_node_label = str(node.attribute)
                dot.node(str(id(node)), label=current_node_label)

                if parent_node:
                    dot.edge(str(id(parent_node)), str(id(node)), label=edge_label)

                for value, child_node in node.children.items():
                    build_tree(child_node, node, value)
            elif isinstance(node, _LeafNode):
                current_node_label = f"Class: {node.label}, Weight: {node.weight}"
                dot.node(str(id(node)), label=current_node_label, shape="box")

                if parent_node:
                    dot.edge(str(id(parent_node)), str(id(node)), label=edge_label)

        build_tree(self.tree)
        dot.format = 'png'
        return dot.render(filename, view=False)

    def print_rules(self, tree=None, rule=''):
        if self.tree is None:
            raise Exception('Decision tree has not been trained yet!')

        if tree is None:
            tree = self.tree
        if rule != '':
            rule += ' AND '
        if isinstance(tree, _LeafNode):
            print(rule[:-3] + ' => ' + tree.label)
            return

        attribute = tree.attribute
        for value, child_node in tree.children.items():
            self.print_rules(child_node, rule + attribute + ' = ' + str(value))

    def rules(self):
        rules = []

        def build_rules(node, parent_node=None, edge_label=None, rule=''):
            if isinstance(node, _DecisionNode):
                current_node_label = node.attribute
                if parent_node:
                    rule += f" AND {current_node_label} = {edge_label}"
                for value, child_node in node.children.items():
                    build_rules(child_node, node, value, rule)
            elif isinstance(node, _LeafNode):
                current_node_label = f"Class: {node.label}, Weight: {node.weight}"
                if parent_node:
                    rule += f" => {current_node_label}"
                rules.append(rule[5:])
        build_rules(self.tree)
        return rules

    def summary(self):
        print("Decision Tree Classifier Summary")
        print("================================")
        print("Number of Instances   : ", len(self.data))
        print("Number of Attributes  : ", len(self.attributes))
        print("Number of Leaves      : ", self.tree.count_leaves())
        print("Number of Rules       : ", len(self.rules()))
        print("Tree Depth            : ", self.tree.depth())

    def get_depth(self):
        if self.tree is None:
            raise Exception('Decision tree has not been trained yet!')
        return self.tree.depth()

    def get_num_nodes(self):
        if self.tree is None:
            raise Exception('Decision tree has not been trained yet!')
        return self.tree.count_nodes()

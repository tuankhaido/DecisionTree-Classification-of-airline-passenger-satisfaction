import math
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class _DecisionNode:
    def __init__(self, attribute, threshold=None):
        self.attribute = attribute
        self.threshold = threshold
        self.children = {}

    def depth(self):
        if len(self.children) == 0:
            return 1
        else:
            return 1 + max(child.depth() for child in self.children.values() if isinstance(child, _DecisionNode))

    def add_child(self, value, node):
        self.children[value] = node

    def count_leaves(self):
        if len(self.children) == 0:
            return 1
        else:
            return sum(child.count_leaves() for child in self.children.values() if isinstance(child, _DecisionNode))


class _LeafNode:
    def __init__(self, label, weight):
        self.label = label
        self.weight = weight


class C45Classifier:
    def __init__(self):
        self.tree = None
        self.attributes = None
        self.data = None
        self.weight = 1

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

        entropy = 0.0

        for count in class_counts.values():
            probability = count / total_weight
            entropy -= probability * math.log2(probability)

        return entropy

    def __split_data(self, data, attribute_index, attribute_value, weights, threshold=None):
        split_data = []
        split_weights = []

        for i, record in enumerate(data):
            if threshold is not None:
                if record[attribute_index] <= threshold:
                    split_data.append(record)
                    split_weights.append(weights[i])
            else:
                if record[attribute_index] == attribute_value:
                    split_data.append(record[:attribute_index] + record[attribute_index+1:])
                    split_weights.append(weights[i])

        return split_data, split_weights

    def __select_best_attribute_c50(self, data, attributes, weights):
        total_entropy = self.__calculate_entropy(data, weights)
        best_attribute = None
        best_gain_ratio = 0.0
        best_threshold = None

        for attribute_index in range(len(attributes)):
            attribute_values = set(record[attribute_index] for record in data)
            attribute_entropy = 0.0
            split_info = 0.0

            if pd.api.types.is_numeric_dtype(data[0][attribute_index]):
                threshold, gain = self.__find_best_threshold(data, attribute_index, weights)
                if gain > best_gain_ratio:
                    best_gain_ratio = gain
                    best_attribute = attribute_index
                    best_threshold = threshold
            else:
                for value in attribute_values:
                    subset, subset_weights = self.__split_data(data, attribute_index, value, weights)
                    subset_entropy = self.__calculate_entropy(subset, subset_weights)
                    subset_probability = sum(subset_weights) / sum(weights)
                    attribute_entropy += subset_probability * subset_entropy
                    if subset_probability > 0:
                        split_info -= subset_probability * math.log2(subset_probability)

                gain = total_entropy - attribute_entropy

                if split_info != 0.0:
                    gain_ratio = gain / split_info
                else:
                    gain_ratio = 0.0

                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_attribute = attribute_index
                    best_threshold = None

        return best_attribute, best_threshold

    def __find_best_threshold(self, data, attribute_index, weights):
        sorted_data = sorted(data, key=lambda x: x[attribute_index])
        best_threshold = None
        best_gain = -float('inf')

        for i in range(1, len(sorted_data)):
            if sorted_data[i][-1] != sorted_data[i - 1][-1]:
                threshold = (sorted_data[i][attribute_index] + sorted_data[i - 1][attribute_index]) / 2
                gain = self.__calculate_info_gain(data, attribute_index, threshold, weights)
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold

        return best_threshold, best_gain

    def __calculate_info_gain(self, data, attribute_index, threshold, weights):
        total_entropy = self.__calculate_entropy(data, weights)

        left_split, left_weights = self.__split_data(data, attribute_index, None, weights, threshold)
        right_split, right_weights = self.__split_data(data, attribute_index, None, weights, threshold + 1)

        left_entropy = self.__calculate_entropy(left_split, left_weights)
        right_entropy = self.__calculate_entropy(right_split, right_weights)

        left_weight = sum(left_weights) / sum(weights)
        right_weight = sum(right_weights) / sum(weights)

        weighted_entropy = left_weight * left_entropy + right_weight * right_entropy

        info_gain = total_entropy - weighted_entropy

        return info_gain

    def __majority_class(self, data, weights):
        class_counts = {}

        for i, record in enumerate(data):
            label = record[-1]
            weight = weights[i]

            if label not in class_counts:
                class_counts[label] = 0.0
            class_counts[label] += weight

        majority_class = max(class_counts, key=class_counts.get)
        return majority_class

    def __build_decision_tree(self, data, attributes, weights, current_depth=0, max_depth=None):
        class_labels = set(record[-1] for record in data)

        if len(class_labels) == 1:
            return _LeafNode(class_labels.pop(), sum(weights))

        if len(attributes) == 0 or (max_depth is not None and current_depth >= max_depth):
            return _LeafNode(self.__majority_class(data, weights), sum(weights))

        best_attribute, best_threshold = self.__select_best_attribute_c50(data, attributes, weights)

        if best_attribute is None:
            return _LeafNode(self.__majority_class(data, weights), sum(weights))

        best_attribute_name = attributes[best_attribute]
        tree = _DecisionNode(best_attribute_name, best_threshold)
        remaining_attributes = attributes[:best_attribute] + attributes[best_attribute+1:]

        if best_threshold is not None:
            left_subset, left_weights = self.__split_data(data, best_attribute, None, weights, best_threshold)
            right_subset, right_weights = self.__split_data(data, best_attribute, None, weights, best_threshold + 1)
            tree.add_child('<=', self.__build_decision_tree(left_subset, remaining_attributes, left_weights, current_depth + 1, max_depth))
            tree.add_child('>', self.__build_decision_tree(right_subset, remaining_attributes, right_weights, current_depth + 1, max_depth))
        else:
            attribute_values = set(record[best_attribute] for record in data)
            for value in attribute_values:
                subset, subset_weights = self.__split_data(data, best_attribute, value, weights)
                if len(subset) == 0:
                    tree.add_child(value, _LeafNode(self.__majority_class(data, weights), sum(subset_weights)))
                else:
                    tree.add_child(value, self.__build_decision_tree(subset, remaining_attributes, subset_weights, current_depth + 1, max_depth))

        return tree

    def __make_tree(self, data, attributes, weights, max_depth=None):
        return self.__build_decision_tree(data, attributes, weights, current_depth=0, max_depth=max_depth)

    def __train(self, data, weight=1, max_depth=None):
        self.weight = weight
        self.attributes = data.columns.tolist()[:-1]
        weights = [self.weight] * len(data)
        self.tree = self.__make_tree(data.values.tolist(), self.attributes, weights, max_depth=max_depth)
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
        attribute_value = instance[attribute_index]

        if tree.threshold is not None:
            if attribute_value <= tree.threshold:
                return self.__classify(tree.children['<='], instance)
            else:
                return self.__classify(tree.children['>'], instance)
        else:
            if attribute_value in tree.children:
                child_node = tree.children[attribute_value]
                return self.__classify(child_node, instance)
            else:
                class_labels = [child.label for child in tree.children.values() if isinstance(child, _LeafNode)]
                if len(class_labels) == 0:
                    return self.__majority_class(self.data.values.tolist(), [1.0] * len(self.data))
                return max(set(class_labels), key=class_labels.count)

    def fit(self, data, label, weight=1, min_entropy=0.1, max_depth=None):
        if isinstance(data, pd.DataFrame):
            data = pd.concat([data, label], axis=1)
        else:
            data = pd.DataFrame(np.c_[data, label])
        self.__train(data, weight, max_depth)

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

    def generate_tree_diagram(self, graphviz, filename):
        dot = graphviz.Digraph()

        def build_tree(node, parent_node=None, edge_label=None):
            if isinstance(node, _DecisionNode):
                current_node_label = str(node.attribute)
                if node.threshold is not None:
                    current_node_label += f" <= {node.threshold}"
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
        if tree.threshold is not None:
            self.print_rules(tree.children['<='], rule + f"{attribute} <= {tree.threshold}")
            self.print_rules(tree.children['>'], rule + f"{attribute} > {tree.threshold}")
        else:
            for value, child_node in tree.children.items():
                self.print_rules(child_node, rule + attribute + ' = ' + str(value))

    def rules(self):
        rules = []

        def build_rules(node, parent_node=None, edge_label=None, rule=''):
            if isinstance(node, _DecisionNode):
                current_node_label = node.attribute
                if node.threshold is not None:
                    current_node_label += f" <= {node.threshold}"
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

    def discretize_data(self, data, attribute):
        # Sort the data by the attribute
        sorted_data = data.sort_values(by=attribute)
        sorted_values = sorted_data[attribute].values
        sorted_labels = sorted_data.iloc[:, -1].values

        # Identify potential thresholds
        potential_thresholds = []
        for i in range(1, len(sorted_values)):
            if sorted_labels[i] != sorted_labels[i - 1]:
                potential_thresholds.append((sorted_values[i] + sorted_values[i - 1]) / 2)

        # Calculate Information Gain for each threshold
        best_threshold = None
        best_info_gain = -float('inf')
        for threshold in potential_thresholds:
            info_gain = self.__calculate_info_gain(sorted_data, attribute, threshold)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_threshold = threshold

        # Create a new binary attribute based on the best threshold
        binary_attribute = f"{attribute}>{best_threshold}"
        data[binary_attribute] = data[attribute] > best_threshold

        return data

    def __calculate_info_gain(self, data, attribute, threshold):
        total_entropy = self.__calculate_entropy(data.values.tolist(), [1] * len(data))

        # Split data based on the threshold
        left_split = data[data[attribute] <= threshold]
        right_split = data[data[attribute] > threshold]

        # Calculate weighted entropy of the splits
        left_entropy = self.__calculate_entropy(left_split.values.tolist(), [1] * len(left_split))
        right_entropy = self.__calculate_entropy(right_split.values.tolist(), [1] * len(right_split))

        left_weight = len(left_split) / len(data)
        right_weight = len(right_split) / len(data)

        weighted_entropy = left_weight * left_entropy + right_weight * right_entropy

        # Information Gain
        info_gain = total_entropy - weighted_entropy

        return info_gain

    def post_prune(self, tree, validation_data):
        if isinstance(tree, _LeafNode):
            return tree

        # Prune subtrees first
        for value, child in tree.children.items():
            tree.children[value] = self.post_prune(child, validation_data)

        # Evaluate the error rate of the current tree
        error_no_prune = self.__calculate_error(tree, validation_data)

        # Convert the current node to a leaf node
        leaf = self.__create_leaf_node(tree)

        # Evaluate the error rate of the pruned tree
        error_prune = self.__calculate_error(leaf, validation_data)

        # Decide whether to prune or not
        if error_prune <= error_no_prune:
            return leaf
        else:
            return tree

    def __calculate_error(self, node, validation_data):
        predictions = self.__classify_node(validation_data.values.tolist(), node)
        actual_labels = validation_data.iloc[:, -1].values
        error = sum(predictions != actual_labels) / len(validation_data)
        return error

    def __create_leaf_node(self, tree):
        most_common_class = self.__find_most_common_class(tree)
        return _LeafNode(most_common_class, sum(tree.children.values()))

    def __find_most_common_class(self, tree):
        class_counts = {}
        for value, child in tree.children.items():
            if isinstance(child, _LeafNode):
                if child.label not in class_counts:
                    class_counts[child.label] = 0
                class_counts[child.label] += child.weight
            else:
                child_class = self.__find_most_common_class(child)
                if child_class not in class_counts:
                    class_counts[child_class] = 0
                class_counts[child_class] += sum(child.children.values())
        return max(class_counts, key=class_counts.get)

    def __classify_node(self, data, node):
        if isinstance(node, _LeafNode):
            return [node.label] * len(data)
        else:
            left_data = [record for record in data if record[self.attributes.index(node.attribute)] <= node.threshold]
            right_data = [record for record in data if record[self.attributes.index(node.attribute)] > node.threshold]
            left_predictions = self.__classify_node(left_data, node.children['<='])
            right_predictions = self.__classify_node(right_data, node.children['>'])
            return left_predictions + right_predictions

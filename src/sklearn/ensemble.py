"""Histogram-based Gradient Boosted Decision Trees in pure NumPy."""
import numpy as np


class _TreeNode:
    __slots__ = ("is_leaf", "value", "feat", "thresh", "left", "right")
    def __init__(self, is_leaf=True, value=0.0, feat=-1, thresh=0.0, left=None, right=None):
        self.is_leaf = is_leaf
        self.value = value
        self.feat = feat
        self.thresh = thresh
        self.left = left
        self.right = right


class HistGradientBoostingClassifier:
    """
    Gradient boosted decision trees using quantile candidate thresholds,
    optimizing binary log-loss with Newton-step leaf weights.
    """
    def __init__(self, max_iter=80, learning_rate=0.15, max_leaf_nodes=31, min_samples_leaf=20, random_state=808):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees_ = []
        self.init_logit_ = 0.0
        self.thresholds_ = []

    def _precompute_thresholds(self, X):
        thresholds = []
        for j in range(X.shape[1]):
            col = X[:, j]
            u = np.unique(np.percentile(col, np.linspace(2, 98, 25)))
            thresholds.append(u)
        return thresholds

    def _build_node(self, X, g, h, depth, max_depth, feat_indices):
        G = float(np.sum(g))
        H = float(np.sum(h))
        leaf_val = -G / (H + 1.0)
        if depth >= max_depth or len(g) < 2 * self.min_samples_leaf:
            return _TreeNode(is_leaf=True, value=leaf_val)

        best_gain = 1e-4
        best_feat = -1
        best_thresh = 0.0
        best_mask = None
        parent_score = (G * G) / (H + 1.0)

        for j in feat_indices:
            col = X[:, j]
            for t in self.thresholds_[j]:
                left_mask = col <= t
                n_left = int(np.sum(left_mask))
                n_right = len(g) - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                G_L = float(np.sum(g[left_mask]))
                H_L = float(np.sum(h[left_mask]))
                G_R = G - G_L
                H_R = H - H_L
                gain = 0.5 * ((G_L * G_L) / (H_L + 1.0) + (G_R * G_R) / (H_R + 1.0) - parent_score)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thresh = float(t)
                    best_mask = left_mask

        if best_feat == -1:
            return _TreeNode(is_leaf=True, value=leaf_val)

        left_child = self._build_node(X[best_mask], g[best_mask], h[best_mask], depth + 1, max_depth, feat_indices)
        right_child = self._build_node(X[~best_mask], g[~best_mask], h[~best_mask], depth + 1, max_depth, feat_indices)
        return _TreeNode(is_leaf=False, value=leaf_val, feat=best_feat, thresh=best_thresh, left=left_child, right=right_child)

    def _predict_tree(self, node, X):
        if node.is_leaf:
            return np.full(X.shape[0], node.value, dtype=float)
        mask = X[:, node.feat] <= node.thresh
        out = np.empty(X.shape[0], dtype=float)
        if np.any(mask):
            out[mask] = self._predict_tree(node.left, X[mask])
        if np.any(~mask):
            out[~mask] = self._predict_tree(node.right, X[~mask])
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = (np.asarray(y) > 0.5).astype(float)
        p_bar = np.clip(np.mean(y), 1e-5, 1.0 - 1e-5)
        self.init_logit_ = float(np.log(p_bar / (1.0 - p_bar)))
        self.thresholds_ = self._precompute_thresholds(X)

        raw_pred = np.full(len(y), self.init_logit_, dtype=float)
        max_depth = 4
        n_feats = X.shape[1]
        feat_idx = np.arange(n_feats)

        self.trees_ = []
        n_rounds = min(self.max_iter, 40)
        for _ in range(n_rounds):
            prob = 1.0 / (1.0 + np.exp(-np.clip(raw_pred, -20.0, 20.0)))
            g = prob - y
            h = np.maximum(prob * (1.0 - prob), 1e-6)
            tree = self._build_node(X, g, h, 0, max_depth, feat_idx)
            update = self._predict_tree(tree, X)
            raw_pred += self.learning_rate * update
            self.trees_.append(tree)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        raw = np.full(X.shape[0], self.init_logit_, dtype=float)
        for tree in self.trees_:
            raw += self.learning_rate * self._predict_tree(tree, X)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(raw, -20.0, 20.0)))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(float)

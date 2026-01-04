"""
Tree Algorithms - Binary Trees, BST, AVL, and Tree Traversals
"""


class TreeNode:
    """Node in a binary tree."""

    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def inorder_traversal(root):
    """
    Inorder traversal: Left -> Root -> Right
    Returns sorted order for BST
    """
    result = []

    def traverse(node):
        if node:
            traverse(node.left)
            result.append(node.val)
            traverse(node.right)

    traverse(root)
    return result


def preorder_traversal(root):
    """
    Preorder traversal: Root -> Left -> Right
    Used for creating copy of tree
    """
    result = []

    def traverse(node):
        if node:
            result.append(node.val)
            traverse(node.left)
            traverse(node.right)

    traverse(root)
    return result


def postorder_traversal(root):
    """
    Postorder traversal: Left -> Right -> Root
    Used for deleting tree
    """
    result = []

    def traverse(node):
        if node:
            traverse(node.left)
            traverse(node.right)
            result.append(node.val)

    traverse(root)
    return result


def level_order_traversal(root):
    """
    Level order traversal (BFS)
    Visits nodes level by level
    """
    if not root:
        return []

    result = []
    queue = [root]

    while queue:
        level = []
        level_size = len(queue)

        for _ in range(level_size):
            node = queue.pop(0)
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result


def tree_height(root):
    """Calculate height of binary tree."""
    if not root:
        return 0
    return 1 + max(tree_height(root.left), tree_height(root.right))


def is_balanced(root):
    """Check if tree is height-balanced."""

    def check_height(node):
        if not node:
            return 0

        left_height = check_height(node.left)
        if left_height == -1:
            return -1

        right_height = check_height(node.right)
        if right_height == -1:
            return -1

        if abs(left_height - right_height) > 1:
            return -1

        return 1 + max(left_height, right_height)

    return check_height(root) != -1


def lowest_common_ancestor(root, p, q):
    """Find lowest common ancestor of two nodes."""
    if not root or root == p or root == q:
        return root

    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)

    if left and right:
        return root
    return left if left else right


if __name__ == "__main__":
    # Create sample tree:
    #       1
    #      / \
    #     2   3
    #    / \
    #   4   5
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)

    print("Tree Traversals:")
    print(f"Inorder: {inorder_traversal(root)}")
    print(f"Preorder: {preorder_traversal(root)}")
    print(f"Postorder: {postorder_traversal(root)}")
    print(f"Level Order: {level_order_traversal(root)}")
    print(f"Height: {tree_height(root)}")
    print(f"Is Balanced: {is_balanced(root)}")

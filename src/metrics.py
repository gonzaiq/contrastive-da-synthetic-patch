import numpy as np

def mmd(X1, X2):
    """
    Compute Maximum Mean Discrepancy (MMD) between two sets X1 and X2 using the expected feature mappings.

    Args:
        X1 (numpy.ndarray): Data from distribution 1, shape (n_samples1, n_features).
        X2 (numpy.ndarray): Data from distribution 2, shape (n_samples2, n_features).

    Returns:
        float: MMD value between the two datasets.
    """
    mean_X1 = np.mean(X1, axis=0)
    mean_X2 = np.mean(X2, axis=0)

    mmd_value = np.sum((mean_X1 - mean_X2) ** 2)

    return mmd_value


def class_wise_mmd(X1, X2, y1, y2):
    """
    Compute class-wise MMD for two labeled domains.

    Args:
        X1 (numpy.ndarray): Data from domain 1, shape (n_samples1, n_features).
        X2 (numpy.ndarray): Data from domain 2, shape (n_samples2, n_features).
        y1 (numpy.ndarray): Labels for domain 1, shape (n_samples1,).
        y2 (numpy.ndarray): Labels for domain 2, shape (n_samples2,).

    Returns:
        float: CMMD value.
    """
    classes = np.union1d(np.unique(y1), np.unique(y2))
    mmd_scores = []

    for cls in classes:
        X1_cls = X1[y1 == cls]
        X2_cls = X2[y2 == cls]

        if len(X1_cls) > 0 and len(X2_cls) > 0:
            mmd_scores.append(mmd(X1_cls, X2_cls))
        else:
            raise ValueError(f"Class {cls} not found in one of the two domains.")

    return np.mean(mmd_scores)

def different_class_mmd(X1, X2, y1, y2, p0=0.5):
    """
    Compute DCMMD for two labeled domains..

    Args:
        X1 (numpy.ndarray): Data from domain 1, shape (n_samples1, n_features).
        X2 (numpy.ndarray): Data from domain 2, shape (n_samples2, n_features).
        y1 (numpy.ndarray): Labels for domain 1, shape (n_samples1,).
        y2 (numpy.ndarray): Labels for domain 2, shape (n_samples2,).
        p0 (float): domain mixture probability.

    Returns:
        float: DCMMD value.
    """
    classes = np.union1d(np.unique(y1), np.unique(y2))
    num_classes = len(classes)

    # Initialize DCMMD value
    dcmmd_value = 0.0

    # Iterate over all class pairs (C1, C2) with C1 != C2
    for _, C1 in enumerate(classes):
        for _, C2 in enumerate(classes):

            if C1 == C2:
                continue

            # Iterate over all domain pairs
            for D1 in [1, 2]:
                for D2 in [1, 2]:

                    # Filter data based on class labels
                    if D1 == 1:
                        X_C1 = X1[y1 == C1]
                        pd1 = p0
                    else:
                        X_C1 = X2[y2 == C1]
                        pd1 = 1 - p0

                    if D2 == 1:
                        X_C2 = X1[y1 == C2]
                        pd2 = 1 - p0
                    else:
                        X_C2 = X2[y2 == C2]
                        pd2 = p0

                    if len(X_C1) == 0:
                        raise ValueError(f"No elements were found for class {C1} and domain {D1}")
                    if len(X_C2) == 0:
                        raise ValueError(f"No elements were found for class {C2} and domain {D2}")

                    # Compute MMD for the selected subsets and normalize it by the mixture probability
                    mmd_value = pd1 * pd2 * mmd(X_C1, X_C2)
                    dcmmd_value += mmd_value

    return dcmmd_value / num_classes

# Example usage:
if __name__ == "__main__":
    # Synthetic data
    np.random.seed(42)
    X1 = np.random.normal(0, 1, (100, 2))
    X2 = np.random.normal(0.5, 1, (100, 2))

    y1 = np.random.choice([0, 1], size=100)
    y2 = np.random.choice([0, 1], size=100)

    mmd_results = mmd(X1, X2)
    print("MMD results:", mmd_results)

    cmmd_results = class_wise_mmd(X1, X2, y1, y2)
    print("Class-wise MMD results:", cmmd_results)

    dcmmd_results = different_class_mmd(X1, X1, y1, y1)
    print("Different class MMD results:", dcmmd_results)

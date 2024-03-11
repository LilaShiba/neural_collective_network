import collections
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from typing import Tuple, List, Union


def degree_distribution(
    G: nx.Graph,
    direction: str,
    scale: int
) -> Union[List[float], Tuple[float, float], Tuple[List[float], Tuple[float, float]]]:
    """
    Analyze and plot the degree distribution of a graph in various ways.

    Parameters:
    - G (nx.Graph): The graph to analyze.
    - direction (str): 'in', 'out', or 'all' to specify the direction of the degrees.
    - scale (int): Specifies the type of scaling and binning to apply.
        0 -> linear scale
        1 -> linear binning with log scale
        2 -> log binning
        3 -> cumulative distribution
        4 -> z-scores

    Returns:
    A list of probabilities, or a tuple with additional information, depending on the scale parameter.
    """
    if direction == 'out':
        degrees = sorted([G.out_degree(n) for n in G.nodes()])
    elif direction == 'in':
        degrees = sorted([G.in_degree(n) for n in G.nodes()])
    else:
        degrees = sorted([G.degree(n) for n in G.nodes()])

    n = G.number_of_nodes()
    degrees_count = collections.Counter(degrees)
    deg, cnt = zip(*degrees_count.items())
    prob_vector = [x / n for x in cnt]

    def linear_scale() -> List[float]:
        """Plot the degree distribution on a linear scale."""
        plt.title(f"Linear Scale {direction}-degree")
        plt.xlabel('K')
        plt.ylabel('P(K)')
        plt.plot(deg, prob_vector, 'o')
        plt.show()
        return prob_vector

    def linear_binning() -> List[float]:
        """Plot the degree distribution with linear binning on a log-log scale."""
        plt.title(f"Linear Binning {direction}-degree")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel('K')
        plt.ylabel('P(K)')
        plt.plot(deg, prob_vector, 'o')
        plt.show()
        return prob_vector

    def log_binning() -> Tuple[float, float]:
        """Plot the degree distribution with log binning."""
        in_max, in_min = max(prob_vector), min(prob_vector)
        log_bins = np.logspace(np.log10(in_min), np.log10(in_max))
        deg_hist, log_bin_edges = np.histogram(
            prob_vector, bins=log_bins, density=True, range=(in_min, in_max))
        plt.title(f"Log Binning & Scaling {direction}-degree")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel('K')
        plt.ylabel('P(K)')
        plt.plot(deg_hist, log_bin_edges[:-1], 'o')
        plt.show()
        return in_min, in_max

    def cum_distro():
        """Plot the cumulative distribution of the degrees."""
        values = np.array(prob_vector)
        cdf = values.cumsum() / values.sum()
        ccdf = 1 - cdf
        plt.xscale("log")
        plt.yscale("log")
        plt.title(f"Cumulative Distribution {direction}-degree")
        plt.ylabel("P(K) >= K")
        plt.xlabel("K")
        plt.plot(cdf[::-1])
        plt.show()

    def z_scores() -> Tuple[List[float], Tuple[float, float]]:
        """Calculate and plot the z-scores of the degree distribution."""
        sigma, mu = np.std(cnt), np.mean(cnt)
        zscores = [round((x - mu) / sigma, 5) for x in cnt]
        plt.title(f"Degree Z-Score Plot {direction}-degree")
        plt.xlabel('Degree')
        plt.ylabel('Norm % of Nodes')
        plt.plot(deg, zscores, 'o')
        plt.show()
        return zscores, (sigma, mu)

    if scale == 0:
        return linear_scale()
    elif scale == 1:
        return linear_binning()
    elif scale == 2:
        return log_binning()
    elif scale == 3:
        return cum_distro()
    elif scale == 4:
        return z_scores()

import os
import random
import re
from rdflib import Graph, URIRef

def load_graph(file_path: str, format: str = "turtle") -> Graph | None:
    """
    Load an RDF graph from a file.

    Args:
        file_path (str): Path to the RDF file.
        format (str, optional): RDF serialization format. Defaults to "turtle".

    Returns:
        Graph | None: Loaded RDF graph, or None if loading fails.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    g = Graph()
    try:
        g.parse(file_path, format=format)
        print(f"Loaded RDF graph from '{file_path}' with {len(g)} triples.")
        return g
    except Exception as e:
        print(f"Failed to parse RDF file: {e}")
        return None

def explore_node(rdf_graph: Graph, node_value: str) -> None:
    """
    Print all outgoing triples for a given node in the RDF graph.

    Args:
        rdf_graph (Graph): The RDF graph.
        node_value (str): The URI of the node to explore.
    """
    uri = URIRef(node_value)

    found_out = False
    for s, p, o in rdf_graph.triples((uri, None, None)):
        print(f"{s} -- {p} --> {o}")
        found_out = True
    if not found_out:
        print("No outgoing triples.")

def extract_numeric_suffix(uri: URIRef) -> int | float:
    """
    Extract a numeric suffix from a URIRef.

    Args:
        uri (URIRef): The URIRef object.

    Returns:
        int | float: The extracted integer suffix, or float('inf') if not found.
    """
    match = re.search(r'/(\d+)$', str(uri))
    return int(match.group(1)) if match else float('inf')
    
if __name__ == "__main__":
    rdf_file = "results/rdf_processed/personas-db.ttl"
    graph = load_graph(rdf_file)

    if graph:
        # subjects = sorted({s for s in graph.subjects() if isinstance(s, URIRef)})
        subjects = sorted({s for s in graph.subjects() if isinstance(s, URIRef)}, key=extract_numeric_suffix)
        total_subjects = len(subjects)

        if total_subjects == 0:
            print("No URIRef subjects found in the graph.")
        else:
            indices_of_interest = [2, 46, 67, 6484] # None or set to sample indices

            valid_indices = None
            if indices_of_interest:
                valid_indices = [i for i in indices_of_interest if 0 <= i < total_subjects]

            if valid_indices:
                selected_subjects = [s for s in subjects if extract_numeric_suffix(s) in valid_indices]
            else:
                print("No valid indices provided. Using random subjects instead.")
                selected_subjects = random.sample(subjects, min(3, total_subjects))

            print("\nExploring selected nodes:")
            for subject in selected_subjects:
                print(f"\nExploring node: {subject}")
                explore_node(graph, str(subject))
                print("\n" + "=" * 200)

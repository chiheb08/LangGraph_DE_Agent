from agent import MediumAgent
import networkx as nx
import matplotlib.pyplot as plt
from IPython.display import Image, display

if __name__ == "__main__":
    # Example URL for testing
    test_url = "https://mayursurani.medium.com/top-20-pyspark-functions-every-data-engineer-should-master-84d3f9640138"
    agent = MediumAgent()
    # Removed visualization as it's handled in the Jupyter notebook
    result = agent.run(test_url)
    if result:
        print("Classification:", result["classification"])
        print("Entities:", ", ".join(result["entities"]))
        print("Summary:", result["summary"])
        print("References:", ", ".join(result["references"]))
    else:
        print("Failed to analyze the article.") 
import os
from typing import List, Dict
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from utils import fetch_medium_articles
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from typing import TypedDict
from IPython.display import Image,display


load_dotenv()

class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str
    references: List[str]

class MediumAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.workflow = self._build_workflow().compile()

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(State)
        workflow.add_node("classification_node", self.classification_node)
        workflow.add_node("entity_extraction", self.entity_extraction_node)
        workflow.add_node("summarization", self.summarize_node)
        workflow.add_node("reference_extraction", self.reference_extraction_node)
        workflow.set_entry_point("classification_node")
        workflow.add_edge("classification_node", "entity_extraction")
        workflow.add_edge("entity_extraction", "summarization")
        workflow.add_edge("summarization", "reference_extraction")
        workflow.add_edge("reference_extraction", END)
        return workflow

    def fetch_article_content(self, url: str) -> str:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Assuming the article content is within <article> tags
            article = soup.find('article')
            return article.get_text(separator=' ', strip=True) if article else ''
        else:
            print(f"Failed to fetch article: {response.status_code}")
            return ''

    def visualize_graph(self):
        try:
            # Generate and display the graph using draw_mermaid_png
            graph_image = self.workflow.get_graph().draw_mermaid_png()
            display(Image(graph_image))
        except Exception as e:
            print(f"Graph visualization requires additional dependencies: {e}")

    def run(self, url: str):
        self.visualize_graph()  # Visualize the graph before running
        article_content = self.fetch_article_content(url)
        if article_content:
            state_input = {"text": article_content}
            result = self.workflow.invoke(state_input)
            return result
        else:
            return None

    def classification_node(self, state: State) -> Dict[str, str]:
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            As a data expert, classify the following text into one of the categories: Technical, Research.

            Text: {text}

            Category:
            """
        )
        message = HumanMessage(content=prompt.format(text=state["text"]))
        classification = self.llm.invoke([message]).content.strip()
        return {"classification": classification}

    def entity_extraction_node(self, state: State) -> Dict[str, List[str]]:
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            As a data expert, extract all technical or buzz words from the following text. Provide the result as a comma-separated list.

            Text: {text}

            Entities:
            """
        )
        message = HumanMessage(content=prompt.format(text=state["text"]))
        entities = self.llm.invoke([message]).content.strip().split(", ")
        return {"entities": entities}

    def summarize_node(self, state: State) -> Dict[str, str]:
        summarization_prompt = PromptTemplate.from_template(
            """
            As a data expert, summarize the following text in one or two sentences.

            Text: {text}

            Summary:
            """
        )
        chain = summarization_prompt | self.llm
        response = chain.invoke({"text": state["text"]})
        return {"summary": response.content}

    def reference_extraction_node(self, state: State) -> Dict[str, List[str]]:
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            As a data expert, extract all references from the following text. Provide the result as a comma-separated list.

            Text: {text}

            References:
            """
        )
        message = HumanMessage(content=prompt.format(text=state["text"]))
        references = self.llm.invoke([message]).content.strip().split(", ")
        return {"references": references} 
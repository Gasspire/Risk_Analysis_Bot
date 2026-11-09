from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from config import MODEL_NAME, TEMPERATURE, BASE_URL
import models
import prompt

class LLMChainFactory:
    """
    Fabbrica per l'inizializzazione modulare delle catene LangChain e dei client LLM.
    Evita istanziazioni globali incontrollate e consente la riconfigurazione dinamica.
    """



    def __init__(
        self,
        model_name: str = MODEL_NAME,
        temperature: float = TEMPERATURE,
        base_url: str = BASE_URL,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = base_url



    def get_llm(self, num_predict: int = 800, json_format: bool = False) -> ChatOllama:
        """Crea un'istanza di ChatOllama configurata."""

        kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
            "base_url": self.base_url,
            "num_ctx": 4096,
            "num_predict": num_predict,
        }
        if json_format:
            kwargs["format"] = "json"

        return ChatOllama(**kwargs)



    def get_classification_chain(self):
        llm = self.get_llm()

        tmpl = ChatPromptTemplate.from_messages([
            ("system", prompt.CLASSIFICATION_SYSTEM_PROMPT),
            ("human", prompt.HUMAN_PROMPT_CLASSIFICATION),
        ]).partial(classification_examples=prompt.CLASSIFICATION_EXAMPLES)

        return tmpl | llm.with_structured_output(models.TypeOfMeasures)



    def get_likelihood_chain(self):
        llm = self.get_llm()

        tmpl = ChatPromptTemplate.from_messages([
            ("system", prompt.LIKELIHOOD_SYSTEM_PROMPT),
            ("human", prompt.HUMAN_PROMPT_LIKELIHOOD),
        ]).partial(likelihood_examples=prompt.LIKELIHOOD_EXAMPLES)
        return tmpl | llm.with_structured_output(models.LikelihoodOutput)



    def get_impact_chain(self):
        llm = self.get_llm()

        tmpl = ChatPromptTemplate.from_messages([
            ("system", prompt.IMPACT_SYSTEM_PROMPT),
            ("human", prompt.HUMAN_PROMPT_IMPACT),
        ]).partial(impact_examples=prompt.IMPACT_EXAMPLES)
        return tmpl | llm.with_structured_output(models.ImpactOutput)



    def get_synthesis_chain(self):
        llm = self.get_llm()

        tmpl = ChatPromptTemplate.from_messages([
            ("system", prompt.SYNTHESIS_SYSTEM_PROMPT),
            ("human", prompt.HUMAN_PROMPT_SYNTHESIS),
        ])
        return tmpl | llm.with_structured_output(models.SynthesisOutput)


    def get_intent_chain(self):
        llm = self.get_llm(num_predict=300)
        tmpl = ChatPromptTemplate.from_messages([
            ("system", prompt.INTENT_SYSTEM_PROMPT),
            ("human", "{input}"),
        ]).partial(intent_examples=prompt.INTENT_EXAMPLES)
        return tmpl | llm.with_structured_output(models.IntentOutput)



    def get_rag_info_chain(self):
        llm = self.get_llm()
        tmpl = ChatPromptTemplate.from_messages([
            ("system", prompt.RAG_INFO_SYSTEM_PROMPT),
            ("human", prompt.HUMAN_PROMPT_RAG_INFO),
        ])
        return tmpl | llm
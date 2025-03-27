# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

from pydantic import BaseModel, Field

from graphrag.config.defaults import graphrag_config_defaults
from graphrag.config.enums import NounPhraseExtractorType


class TextAnalyzerConfig(BaseModel):
    """Configuration section for NLP text analyzer."""

    extractor_type: NounPhraseExtractorType = Field(
        description="The noun phrase extractor type.",
        default=graphrag_config_defaults.extract_graph_nlp.text_analyzer.extractor_type,
    )
    mode: str = Field(
        description="The bert model mode.",
        default=graphrag_config_defaults.extract_graph_nlp.text_analyzer.mode,
    )
    model_dir: str = Field(
        description="The bert model path.",
        default=graphrag_config_defaults.extract_graph_nlp.text_analyzer.model_dir,
    )
    m_model_name: str = Field(
        description="The muilt language bert model name.",
        default=graphrag_config_defaults.extract_graph_nlp.text_analyzer.m_model_name,
    )
    cn_model_name: str = Field(
        description="The chinese bert model name.",
        default=graphrag_config_defaults.extract_graph_nlp.text_analyzer.cn_model_name,
    )
    en_model_name: str = Field(
        description="The english bert model name.",
        default=graphrag_config_defaults.extract_graph_nlp.text_analyzer.en_model_name,
    )
    bio_model_name: str = Field(
        description="The bio bert model name.",
        default=graphrag_config_defaults.extract_graph_nlp.text_analyzer.bio_model_name,
    )
    m_model_device: int = Field(
        description="The device to use in m bert ner.",
        default=graphrag_config_defaults.extract_graph_nlp.text_analyzer.m_model_device,
    )
    cn_model_device: int = Field(
        description="The device to use in cn bert ner.",
        default=graphrag_config_defaults.extract_graph_nlp.text_analyzer.cn_model_device,
    )
    en_model_device: int = Field(
        description="The device to use in en ber ner.",
        default=graphrag_config_defaults.extract_graph_nlp.text_analyzer.cn_model_device,
    )
    bio_model_device: int = Field(
        description="The device to use in bio bert ner.",
        default=graphrag_config_defaults.extract_graph_nlp.text_analyzer.bio_model_device,
    )
    score_threshold: float = Field(
        description="The category score threshold of BERT model NER.",
        default=graphrag_config_defaults.extract_graph_nlp.text_analyzer.score_threshold,
    )
    m_labels: list[str] = Field(
        description="Categories filtered by Muilt Language BERT model NER.",
        default_factory=lambda: graphrag_config_defaults.extract_graph_nlp.text_analyzer.cn_labels,
    )
    cn_labels: list[str] = Field(
        description="Categories filtered by Chinese BERT model NER.",
        default_factory=lambda: graphrag_config_defaults.extract_graph_nlp.text_analyzer.cn_labels,
    )
    en_labels: list[str] = Field(
        description="Categories filtered by English BERT model NER.",
        default=graphrag_config_defaults.extract_graph_nlp.text_analyzer.en_labels,
    )
    bio_labels: list[str] = Field(
        description="Categories filtered by Bio BERT model NER.",
        default_factory=lambda: graphrag_config_defaults.extract_graph_nlp.text_analyzer.bio_labels,
    )
    max_word_length: int = Field(
        description="The max word length for NLP parsing.",
        default=graphrag_config_defaults.extract_graph_nlp.text_analyzer.max_word_length,
    )
    word_delimiter: str = Field(
        description="The delimiter for splitting words.",
        default=graphrag_config_defaults.extract_graph_nlp.text_analyzer.word_delimiter,
    )
    include_named_entities: bool = Field(
        description="Whether to include named entities in noun phrases.",
        default=graphrag_config_defaults.extract_graph_nlp.text_analyzer.include_named_entities,
    )
    exclude_nouns: list[str] | None = Field(
        description="The list of excluded nouns (i.e., stopwords). If None, will use a default stopword list",
        default=graphrag_config_defaults.extract_graph_nlp.text_analyzer.exclude_nouns,
    )
    exclude_entity_tags: list[str] = Field(
        description="The list of named entity tags to exclude in noun phrases.",
        default=graphrag_config_defaults.extract_graph_nlp.text_analyzer.exclude_entity_tags,
    )
    exclude_pos_tags: list[str] = Field(
        description="The list of part-of-speech tags to remove in noun phrases.",
        default=graphrag_config_defaults.extract_graph_nlp.text_analyzer.exclude_pos_tags,
    )
    noun_phrase_tags: list[str] = Field(
        description="The list of noun phrase tags.",
        default=graphrag_config_defaults.extract_graph_nlp.text_analyzer.noun_phrase_tags,
    )
    noun_phrase_grammars: dict[str, str] = Field(
        description="The CFG for matching noun phrases. The key is a tuple of POS tags and the value is the grammar.",
        default=graphrag_config_defaults.extract_graph_nlp.text_analyzer.noun_phrase_grammars,
    )


class ExtractGraphNLPConfig(BaseModel):
    """Configuration section for graph extraction via NLP."""

    normalize_edge_weights: bool = Field(
        description="Whether to normalize edge weights.",
        default=graphrag_config_defaults.extract_graph_nlp.normalize_edge_weights,
    )
    text_analyzer: TextAnalyzerConfig = Field(
        description="The text analyzer configuration.", default=TextAnalyzerConfig()
    )
    concurrent_requests: int = Field(
        description="The number of threads to use for the extraction process.",
        default=graphrag_config_defaults.extract_graph_nlp.concurrent_requests,
    )

# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Create a noun phrase extractor from a configuration."""

from typing import ClassVar

from graphrag.config.enums import NounPhraseExtractorType
from graphrag.config.models.extract_graph_nlp_config import TextAnalyzerConfig
from graphrag.index.operations.build_noun_graph.np_extractors.base import (
    BaseNounPhraseExtractor,
)
from graphrag.index.operations.build_noun_graph.np_extractors.cfg_extractor import (
    CFGNounPhraseExtractor,
)
from graphrag.index.operations.build_noun_graph.np_extractors.regex_extractor import (
    RegexENNounPhraseExtractor,
)
from graphrag.index.operations.build_noun_graph.np_extractors.stop_words import (
    EN_STOP_WORDS,
)
from graphrag.index.operations.build_noun_graph.np_extractors.syntactic_parsing_extractor import (
    SyntacticNounPhraseExtractor,
)


class NounPhraseExtractorFactory:
    """A factory class for creating noun phrase extractor."""

    np_extractor_types: ClassVar[dict[str, type]] = {}

    @classmethod
    def register(cls, np_extractor_type: str, np_extractor: type):
        """Register a vector store type."""
        cls.np_extractor_types[np_extractor_type] = np_extractor

    @classmethod
    def get_np_extractor(cls, config: TextAnalyzerConfig) -> BaseNounPhraseExtractor:
        """Get the noun phrase extractor type from a string."""
        np_extractor_type = config.extractor_type
        exclude_nouns = config.exclude_nouns
        if exclude_nouns is None:
            exclude_nouns = EN_STOP_WORDS
        match np_extractor_type:
            case NounPhraseExtractorType.Syntactic:
                return SyntacticNounPhraseExtractor(
                    mode=config.mode,
                    model_dir=config.model_dir,
                    m_model_name=config.m_model_name,
                    cn_model_name=config.cn_model_name,
                    en_model_name=config.en_model_name,
                    bio_model_name=config.bio_model_name,
                    m_model_device=config.m_model_device,
                    cn_model_device=config.cn_model_device,
                    en_model_device=config.en_model_device,
                    bio_model_device=config.bio_model_device,
                    score_threshold=config.score_threshold,
                    m_labels=config.m_labels,
                    cn_labels=config.cn_labels,
                    en_labels=config.en_labels,
                    bio_labels=config.bio_labels,
                    max_word_length=config.max_word_length,
                    include_named_entities=config.include_named_entities,
                    exclude_entity_tags=config.exclude_entity_tags,
                    # exclude_pos_tags=config.exclude_pos_tags,
                    exclude_nouns=exclude_nouns,
                    word_delimiter=config.word_delimiter,
                )
            case NounPhraseExtractorType.CFG:
                grammars = {}
                for key, value in config.noun_phrase_grammars.items():
                    grammars[tuple(key.split(","))] = value
                return CFGNounPhraseExtractor(
                    model_name=config.model_name,
                    max_word_length=config.max_word_length,
                    include_named_entities=config.include_named_entities,
                    exclude_entity_tags=config.exclude_entity_tags,
                    exclude_pos_tags=config.exclude_pos_tags,
                    exclude_nouns=exclude_nouns,
                    word_delimiter=config.word_delimiter,
                    noun_phrase_grammars=grammars,
                    noun_phrase_tags=config.noun_phrase_tags,
                )
            case NounPhraseExtractorType.RegexEnglish:
                return RegexENNounPhraseExtractor(
                    exclude_nouns=exclude_nouns,
                    max_word_length=config.max_word_length,
                    word_delimiter=config.word_delimiter,
                )


def create_noun_phrase_extractor(
    analyzer_config: TextAnalyzerConfig,
) -> BaseNounPhraseExtractor:
    """Create a noun phrase extractor from a configuration."""
    return NounPhraseExtractorFactory.get_np_extractor(analyzer_config)

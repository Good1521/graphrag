# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Noun phrase extractor based on dependency parsing and NER using SpaCy."""

from typing import Any

from spacy.tokens.span import Span
from spacy.util import filter_spans

from graphrag.index.operations.build_noun_graph.np_extractors.base import (
    BaseNounPhraseExtractor,
)
from graphrag.index.operations.build_noun_graph.np_extractors.np_validator import (
    has_valid_token_length,
    is_compound,
    is_valid_entity,
)


from transformers import BertTokenizer, BertForTokenClassification, pipeline
import re
from langdetect import detect

class SyntacticNounPhraseExtractor(BaseNounPhraseExtractor):
    """基于 BERT 的名词短语提取器，使用命名实体识别 (NER) 和依存句法分析。"""
    
    def __init__(
        self,
        model_name: str,
        max_word_length: int,
        include_named_entities: bool,
        exclude_entity_tags: list[str],
        exclude_nouns: list[str],
        word_delimiter: str,
    ):
        """
        初始化 BERT NER 模型。

        Args:
            model_name: BERT 模型名称。
            max_word_length: 最大单词长度。
            include_named_entities: 是否包含命名实体。
            exclude_entity_tags: 需要排除的实体标签。
            exclude_nouns: 需要排除的名词。
            word_delimiter: 词语之间的连接符。
        """
        super().__init__(
            model_name=model_name,
            max_word_length=max_word_length,
            exclude_nouns=exclude_nouns,
            word_delimiter=word_delimiter,
        )
        self.include_named_entities = include_named_entities
        self.exclude_entity_tags = exclude_entity_tags

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, device=0)
    
    def extract(self, text: str) -> list[str]:
        """提取文本中的名词短语。"""
        doc_chunks = self._chunk_text(text)
        filtered_noun_phrases = set()
        for chunk in doc_chunks:
            entities = self.ner_pipeline(chunk)
            tokenized_length = len(self.tokenizer.tokenize(chunk))
            processed_entities = self._process_entities(entities)

            for entity in processed_entities:
                if entity['entity_group'] not in self.exclude_entity_tags:
                    noun_phrase = entity['word']
                    # 过滤掉 [UNK] 或者类似的无效标记
                    if "[UNK]" in noun_phrase:
                        continue
                    if noun_phrase.upper() not in self.exclude_nouns:
                        filtered_noun_phrases.add(noun_phrase)

        return list(filtered_noun_phrases)
    
    def _chunk_text(self, text: str, max_length: int = 512) -> list[str]:
        lang = detect(text)
        if lang == "zh-cn":
            sentences = re.split(r'([。！？；\n])', text)  # 中文标点
        elif lang == "en":
            sentences = re.split(r'([.!?]\s+)', text)  # 英文标点
        else:
            sentences = re.split(r'([。！？；\n])', text)  # 中文标点

        """将文本拆分为小块，确保每块不超过最大长度。"""
        chunks = []
        current_chunk = []
        current_length = 0

        for i in range(0, len(sentences), 2):
            sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
            tokenized_length = len(self.tokenizer.tokenize(sentence))

            if current_length + tokenized_length > max_length:
                chunks.append("".join(current_chunk))
                current_chunk = [sentence]
                current_length = tokenized_length
            else:
                current_chunk.append(sentence)
                current_length += tokenized_length

        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks
    
    def _process_entities(self, entities: list[dict]) -> list[dict]:
        """合并 BERT 识别的实体，确保整体性。"""
        merged_entities = []
        current_entity = []
        current_label = None
        
        for entity in entities:
            word = entity["word"].replace("##", "")
            label = entity["entity"]

            if label.startswith("B-"):
                if current_entity:
                    merged_entities.append({"entity_group": current_label, "word": "".join(current_entity)})
                current_entity = [word]
                current_label = label[2:]
            elif label.startswith("I-") or label.startswith("E-"):
                current_entity.append(word)
        
        if current_entity:
            merged_entities.append({"entity_group": current_label, "word": "".join(current_entity)})
        
        return merged_entities
    
    def __str__(self) -> str:
        return f"bert_{self.model_name}_{self.max_word_length}_{self.include_named_entities}_{self.exclude_entity_tags}_{self.exclude_nouns}_{self.word_delimiter}"


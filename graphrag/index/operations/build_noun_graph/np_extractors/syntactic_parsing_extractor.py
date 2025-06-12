# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Noun phrase extractor based on dependency parsing and NER using SpaCy."""

from typing import Any
import re, string

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
import os

class SyntacticNounPhraseExtractor(BaseNounPhraseExtractor):
    """基于 BERT 的名词短语提取器，使用命名实体识别 (NER) 和依存句法分析。"""
    
    def __init__(
        self,
        mode: str,
        model_dir: str,
        m_model_name: str,
        cn_model_name: str,
        en_model_name: str,
        bio_model_name: str,
        m_model_device: int,
        cn_model_device: int,
        en_model_device: int,
        bio_model_device: int,
        score_threshold: float,
        m_labels: list[str],
        cn_labels: list[str],
        en_labels: list[str],
        bio_labels: list[str],
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
        m_model_path = os.path.join(model_dir, m_model_name)
        cn_model_path = os.path.join(model_dir, cn_model_name)
        en_model_path = os.path.join(model_dir, en_model_name)
        bio_model_path = os.path.join(model_dir, bio_model_name)

        super().__init__(
            model_name=cn_model_path,
            max_word_length=max_word_length,
            exclude_nouns=exclude_nouns,
            word_delimiter=word_delimiter,
        )
        self.include_named_entities = include_named_entities
        self.exclude_entity_tags = exclude_entity_tags

        self.mode = mode

        if self.mode == "muilt":
            self.m_tokenizer = BertTokenizer.from_pretrained(m_model_path)
            self.m_model = BertForTokenClassification.from_pretrained(m_model_path)
            self.m_ner_pipeline = pipeline("ner", model=self.m_model, tokenizer=self.m_tokenizer, device=m_model_device)
            
            self.m_labels = m_labels

        elif self.mode == "two":
            self.cn_tokenizer = BertTokenizer.from_pretrained(cn_model_path)
            self.cn_model = BertForTokenClassification.from_pretrained(cn_model_path)
            self.cn_ner_pipeline = pipeline("ner", model=self.cn_model, tokenizer=self.cn_tokenizer, device=cn_model_device)

            self.en_tokenizer = BertTokenizer.from_pretrained(en_model_path)
            self.en_model = BertForTokenClassification.from_pretrained(en_model_path)
            self.en_ner_pipeline = pipeline("ner", model=self.en_model, tokenizer=self.en_tokenizer, device=en_model_device)
            
            self.cn_labels = cn_labels
            self.en_labels = en_labels      

        elif self.mode == "bio":
            self.bio_tokenizer = BertTokenizer.from_pretrained(bio_model_path)
            self.bio_model = BertForTokenClassification.from_pretrained(bio_model_path)
            self.bio_ner_pipeline = pipeline("ner", model=self.bio_model, tokenizer=self.bio_tokenizer, device=bio_model_device)
            
            self.bio_labels = bio_labels

       

        self.score_threshold = score_threshold


    def extract(self, text: str) -> list[str]:
        lang_text = detect(text)

        if self.mode == "muilt":
            tokenizer = self.m_tokenizer
            max_length = self.m_model.config.max_position_embeddings

        elif self.mode == "two":
            if lang_text == "zh-cn":
                tokenizer = self.cn_tokenizer
                max_length = self.cn_model.config.max_position_embeddings
            elif lang_text == "en":
                tokenizer = self.en_tokenizer
                max_length = self.en_model.config.max_position_embeddings
            else:
                tokenizer = self.cn_tokenizer
                max_length = self.cn_model.config.max_position_embeddings

        elif self.mode == "bio":
            tokenizer = self.bio_tokenizer
            max_length = self.bio_model.config.max_position_embeddings


        """提取文本中的名词短语。"""
        doc_chunks = self._chunk_text(tokenizer, lang_text, text, max_length)
        filtered_noun_phrases = set()
        if self.mode == "muilt":
            ner_pipeline = self.m_ner_pipeline
            all_entities = ner_pipeline(doc_chunks)
            all_chunks = doc_chunks
        elif self.mode == "two":
            lang_detected_chunks = [detect(chunk) for chunk in doc_chunks]
            zh_chunks = [chunk for chunk, lang in zip(doc_chunks, lang_detected_chunks) if lang == "zh-cn"]
            en_chunks = [chunk for chunk, lang in zip(doc_chunks, lang_detected_chunks) if lang == "en"]
            other_chunks = [chunk for chunk, lang in zip(doc_chunks, lang_detected_chunks) if lang not in ("zh-cn", "en")]
            
            # 分语言批量推理
            zh_entities = self.cn_ner_pipeline(zh_chunks)
            en_entities = self.en_ner_pipeline(en_chunks)
            other_entities = self.cn_ner_pipeline(other_chunks)

            # 合并为统一序列
            all_chunks = zh_chunks + en_chunks + other_chunks
            all_entities = zh_entities + en_entities + other_entities
        elif self.mode == "bio":
            ner_pipeline = self.bio_ner_pipeline
            all_entities = ner_pipeline(doc_chunks)
            all_chunks = doc_chunks

        for chunk, entities in zip(all_chunks, all_entities):
            if self.mode == "muilt":
                processed_entities = self._muilt_process_entities(self.m_labels, entities)
            elif self.mode == "two":
                lang = detect(chunk)
                if lang == "zh-cn":
                    processed_entities = self._cn_process_entities(self.cn_labels, entities)
                elif lang == "en":
                    processed_entities = self._muilt_process_entities(self.en_labels, entities)
                else:
                    processed_entities = self._cn_process_entities(self.cn_labels, entities)
            elif self.mode == "bio":
                processed_entities = self._muilt_process_entities(self.bio_labels, entities)

            for entity in processed_entities:
                if entity['entity_group'] not in self.exclude_entity_tags:
                    noun_phrase = entity['word']
                    if "[UNK]" in noun_phrase:
                        continue
                    if noun_phrase.upper() not in self.exclude_nouns:
                        filtered_noun_phrases.add(noun_phrase)

        return list(filtered_noun_phrases)

    
    def _chunk_text(self, tokenizer, lang, text: str, max_length: int = 512) -> list[str]:
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
            tokenized_length = len(tokenizer.tokenize(sentence))

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
    
    def _cn_process_entities(self, labels, entities: list[dict]) -> list[dict]:
        """合并 BERT 识别的实体，确保整体性。"""
        merged_entities = []
        current_entity = []
        current_label = None
        current_entity_score = 0
        is_complete_entity = False  # 用于判断是否有 E- 标记
        
        for entity in entities:

            entity_label_name = entity["entity"][2:]

            if labels and entity_label_name not in labels:
                continue  # 如果 cn_entity_list 存在且 label 不在其中，则跳过
            
            word = entity["word"].replace("##", "")
            label = entity["entity"]

            if label.startswith("B-"):  # 实体开始
                # 只有完整实体（有E-）才加入
                if current_entity and is_complete_entity and current_entity_score >= self.score_threshold:
                    merged_entities.append({
                        "entity_group": current_label,
                        "word": "".join(current_entity)
                    })

                # 开始新实体
                current_entity = [word]
                current_label = label[2:]  # 去掉 "B-" 前缀
                current_entity_score = entity["score"]
                is_complete_entity = False  # 重新标记完整性

            elif label.startswith("I-") and current_label == label[2:]:
                current_entity.append(word)
                current_entity_score = (current_entity_score + entity["score"]) / 2

            elif label.startswith("E-") and current_label == label[2:]:
                current_entity.append(word)
                current_entity_score = (current_entity_score + entity["score"]) / 2
                is_complete_entity = True  # 标记为完整实体

        
        # 确保最后一个实体完整后才加入
        if current_entity and is_complete_entity and current_entity_score >= self.score_threshold:
            merged_entities.append({
                "entity_group": current_label,
                "word": "".join(current_entity)
            })

        return merged_entities
    
    def _en_process_entities(self, labels, entities: list[dict]) -> list[dict]:
        """合并 BERT 识别的实体，确保整体性。"""
        merged_entities = []
        current_entity = []
        current_label = None
        current_entity_score = 0
        
        for entity in entities:

            entity_label_name = entity["entity"][2:]
            if labels and entity_label_name not in labels:
                continue  # 如果 cn_entity_list 存在且 label 不在其中，则跳过
            
            word = entity["word"].replace("##", "")
            label = entity["entity"]

            if label.startswith("B-"):  # 新实体开始
                if current_entity and current_entity_score >= self.score_threshold:  # 先保存之前的实体
                    merged_entities.append({
                        "entity_group": current_label,
                        "word": "".join(current_entity)
                    })
                # 开始新实体
                current_entity = [word]
                current_label = label[2:]  # 去掉 "B-" 前缀
                current_entity_score = entity["score"]

            elif label.startswith("I-"):  # 继续当前实体
                current_entity.append(word)
                current_entity_score = (current_entity_score + entity["score"]) / 2

        
        if current_entity and current_entity_score >= self.score_threshold:
            merged_entities.append({
                "entity_group": current_label,
                "word": "".join(current_entity)
            })

        return merged_entities

    def is_chinese(self, text):
        """ 判断字符串是否包含中文字符 """
        flag = bool(re.search(r'[\u4e00-\u9fa5]', text))
        return flag

    def is_word(self, text):
        flag = not text.startswith("##")
        return flag
    
    def _muilt_process_entities(self, labels, entities: list[dict]) -> list[dict]:
        """合并 BERT 识别的实体，确保整体性。"""
        merged_entities = []
        current_entity = []
        current_label = None
        current_entity_score = 0
        
        for entity in entities:

            entity_label_name = entity["entity"][2:]
            if labels and entity_label_name not in labels:
                continue  # 如果 cn_entity_list 存在且 label 不在其中，则跳过
            
            word = entity["word"]
            label = entity["entity"]

            if "UNK" in word:
                continue

            if label.startswith("B-") and word.startswith("##"):
                label = "I-" + label[2:]

            if label.startswith("B-"):  # 新实体开始
                if current_entity and current_entity_score >= self.score_threshold:  # 先保存之前的实体
                    if self.is_chinese("".join(current_entity)) and self.is_word("".join(current_entity)):
                        merged_entities.append({
                            "entity_group": current_label,
                            "word": "".join(current_entity)
                        })
                    else:
                        merged_entities.append({
                            "entity_group": current_label,
                            "word": " ".join(current_entity)
                        })

                # 开始新实体
                current_entity = [word]
                current_label = label[2:]  # 去掉 "B-" 前缀
                current_entity_score = entity["score"]

            elif label.startswith("I-") and current_label == label[2:]:  # 继续当前实体
                # current_entity.append(word)
                if not word.startswith("##"):
                    if word in string.punctuation:
                        # 只在 current_entity 不为空时修改最后一个元素
                        if current_entity:
                            current_entity[-1] = current_entity[-1] + word
                        else:
                            # 这里可以选择怎么处理，如果没有实体（比如遇到错误输入或特殊情况），可以选择跳过
                            current_entity.append(word.replace("##", ""))  # 或者设置为空字符串，或直接跳过
                    else:
                        current_entity.append(word)
                else:
                    # 只在 current_entity 不为空时修改最后一个元素
                    if current_entity:
                        current_entity[-1] = current_entity[-1] + word.replace("##", "")
                    else:
                        # 这里可以选择怎么处理，如果没有实体（比如遇到错误输入或特殊情况），可以选择跳过
                        current_entity.append(word.replace("##", ""))  # 或者设置为空字符串，或直接跳过
                
                current_entity_score = (current_entity_score + entity["score"]) / 2

        
        if current_entity and current_entity_score >= self.score_threshold:
            if self.is_chinese("".join(current_entity)) and self.is_word("".join(current_entity)):
                merged_entities.append({
                    "entity_group": current_label,
                    "word": "".join(current_entity)
                })
            else:
                merged_entities.append({
                    "entity_group": current_label,
                    "word": " ".join(current_entity)
                })

        return merged_entities
    
    def __str__(self) -> str:
        return f"bert_{self.model_name}_{self.max_word_length}_{self.include_named_entities}_{self.exclude_entity_tags}_{self.exclude_nouns}_{self.word_delimiter}"


# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
Indexing API for GraphRAG.

WARNING: This API is under development and may undergo changes in future releases.
Backwards compatibility is not guaranteed at this time.
"""

import logging

from graphrag.callbacks.reporting import create_pipeline_reporter
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.enums import IndexingMethod
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.run.run_pipeline import run_pipeline
from graphrag.index.run.utils import create_callback_chain
from graphrag.index.typing.pipeline_run_result import PipelineRunResult
from graphrag.index.typing.workflow import WorkflowFunction
from graphrag.index.workflows.factory import PipelineFactory
from graphrag.logger.base import ProgressLogger
from graphrag.logger.null_progress import NullProgressLogger

from pathlib import Path
import argparse
import asyncio 
import sys

log = logging.getLogger(__name__)


async def build_index(
    download_task,
    config: GraphRagConfig,
    method: IndexingMethod = IndexingMethod.Standard,
    is_update_run: bool = False,
    memory_profile: bool = False,
    callbacks: list[WorkflowCallbacks] | None = None,
    progress_logger: ProgressLogger | None = None,
) -> list[PipelineRunResult]:
    """Run the pipeline with the given configuration.

    Parameters
    ----------
    config : GraphRagConfig
        The configuration.
    method : IndexingMethod default=IndexingMethod.Standard
        Styling of indexing to perform (full LLM, NLP + LLM, etc.).
    memory_profile : bool
        Whether to enable memory profiling.
    callbacks : list[WorkflowCallbacks] | None default=None
        A list of callbacks to register.
    progress_logger : ProgressLogger | None default=None
        The progress logger.

    Returns
    -------
    list[PipelineRunResult]
        The list of pipeline run results
    """
    logger = progress_logger or NullProgressLogger()
    # create a pipeline reporter and add to any additional callbacks
    callbacks = callbacks or []
    callbacks.append(create_pipeline_reporter(config.reporting, None))

    workflow_callbacks = create_callback_chain(callbacks, logger)

    outputs: list[PipelineRunResult] = []

    if memory_profile:
        log.warning("New pipeline does not yet support memory profiling.")

    pipeline = PipelineFactory.create_pipeline(config, method)

    workflow_callbacks.pipeline_start(pipeline.names())

    async for output in run_pipeline(
        download_task,
        pipeline,
        config,
        callbacks=workflow_callbacks,
        logger=logger,
        is_update_run=is_update_run,
    ):
        outputs.append(output)
        if output.errors and len(output.errors) > 0:
            logger.error(output.workflow)
        else:
            logger.success(output.workflow)
        logger.info(str(output.result))

    workflow_callbacks.pipeline_end(outputs)
    return outputs


def register_workflow_function(name: str, workflow: WorkflowFunction):
    """Register a custom workflow function. You can then include the name in the settings.yaml workflows list."""
    PipelineFactory.register(name, workflow)


def _logger(logger: ProgressLogger):
    def info(msg: str, verbose: bool = False):
        log.info(msg)
        if verbose:
            logger.info(msg)

    def error(msg: str, verbose: bool = False):
        log.error(msg)
        if verbose:
            logger.error(msg)

    def success(msg: str, verbose: bool = False):
        log.info(msg)
        if verbose:
            logger.success(msg)

    return info, error, success


def _register_signal_handlers(logger: ProgressLogger):
    import signal

    def handle_signal(signum, _):
        # Handle the signal here
        logger.info(f"Received signal {signum}, exiting...")  # noqa: G004
        logger.dispose()
        for task in asyncio.all_tasks():
            task.cancel()
        logger.info("All tasks cancelled. Exiting...")

    # Register signal handlers for SIGINT and SIGHUP
    signal.signal(signal.SIGINT, handle_signal)

    if sys.platform != "win32":
        signal.signal(signal.SIGHUP, handle_signal)


async def get_index(download_task,root_directory,config_file,method,is_update_run, mode, labels, score_threshold=0.7, device1=0, device2=0):
    from graphrag.config.load_config import load_config
    from graphrag.logger.factory import LoggerFactory, LoggerType
    from graphrag.config.enums import CacheType, IndexingMethod
    from graphrag.config.logging import enable_logging_with_config
    from graphrag.index.validate_config import async_validate_config_names
    from graphrag.utils.cli import redact

    if method == "fast":
        index_method = IndexingMethod.Fast
    elif method == "standard":
        index_method = IndexingMethod.Standard
    else:
        index_method = IndexingMethod.Fast

    # 调用函数加载配置
    config = load_config(Path(root_directory), config_filepath=Path(config_file), cli_overrides={})

    # print("配置文件是", config)
    if mode:
        config.extract_graph_nlp.text_analyzer.mode = mode

    if labels:
        if mode == "muilt":
            config.extract_graph_nlp.text_analyzer.m_labels = labels[mode]
            config.extract_graph_nlp.text_analyzer.m_model_device = int(device1)
        elif mode == "two":
            config.extract_graph_nlp.text_analyzer.cn_labels = labels["cn"]
            config.extract_graph_nlp.text_analyzer.en_labels = labels["en"]
            config.extract_graph_nlp.text_analyzer.cn_model_device = int(device1)
            config.extract_graph_nlp.text_analyzer.en_model_device = int(device2)
        elif mode == "bio":
            config.extract_graph_nlp.text_analyzer.bio_labels = labels[mode]
            config.extract_graph_nlp.text_analyzer.bio_model_device = int(device1)

    config.extract_graph_nlp.text_analyzer.score_threshold = float(score_threshold)

    # print("------------")
    # print("配置文件是", config)
    # """
    logger = LoggerType("none")  # rich, none, print
    progress_logger = LoggerFactory().create_logger(logger)
    info, error, success = _logger(progress_logger)

    cache = True

    if not cache:
        config.cache.type = CacheType.none

    verbose = False
    enabled_logging, log_path = enable_logging_with_config(config, verbose)
    if enabled_logging:
        info(f"Logging enabled at {log_path}", True)
    else:
        info(
            f"Logging not enabled for config {redact(config.model_dump())}",
            True,
        )

    skip_validation = False
    if not skip_validation:
        if_model_work = await async_validate_config_names(progress_logger, config)

    if not if_model_work:
        print("验证大模型发生了错误连接失败------->", if_model_work)
        download_task.task_exception()
        return None
    
    dry_run = False
    info(f"Starting pipeline run. {dry_run=}", verbose)
    info(
        f"Using default configuration: {redact(config.model_dump())}",
        verbose,
    )

    if dry_run:
        info("Dry run complete, exiting...", True)
        sys.exit(0)

    _register_signal_handlers(progress_logger)

    print("开始index")

    # 使用配置
    try:
        outputs = await build_index(
            download_task=download_task,
            config=config,
            method=index_method,
            is_update_run=is_update_run,
            memory_profile=False,
            progress_logger=progress_logger,
        )
    except Exception as e:
        print("graphrag发生了错误------>", e)
        download_task.task_exception()
        sys.exit(1)


    download_task.finish()
    print("重建结束")
    
    return outputs
    # """

# 存储下载任务信息的类
class DownloadTask:
    def __init__(self, session_id: str, progress_queue: asyncio.Queue):
        self.session_id = session_id
        self.workflow_name = None
        self.num_workflows = 0
        self.finished_workflows = 0
        self.now_workflow = 0
        self.progress = 0
        self.progress_queue = progress_queue
        self.is_downloading= None
        self.is_stop = False
        self.is_graceful_stop = None
        self.complete_finish = None
        self.is_exception = False

    def update_mes(self, workflow_name: str, num_workflows: int, finished_workflows: int, now_workflow: int):
        self.workflow_name = workflow_name
        self.num_workflows = num_workflows
        self.finished_workflows = finished_workflows
        self.now_workflow = now_workflow
        self.progress = round(finished_workflows / num_workflows * 100, 2)

    def start(self):
        self.is_downloading = True

    def stop(self):
        self.is_stop = True

    def finish(self):
        self.is_downloading = False

    def task_exception(self):
        self.is_exception = True

    def complete_finished(self):
        self.complete_finish = True

    def graceful_stop(self):
        self.is_graceful_stop = True
        self.complete_finish = False




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, default='/home/turing/graphragtest/graphrag_2.0.0/ragtest')
    parser.add_argument("--config_file", type=str, default='/home/turing/workspace/rag/stores/default_setting/settings_cn.yaml')
    parser.add_argument("--is_update_run", type=str, default=False)
    parser.add_argument("--index_method", type=str, default="fast")
    parser.add_argument("--bert_mode", type=str, default="muilt")
    parser.add_argument("--ner_labels", type=str, default={"muilt": [""]})

    args = parser.parse_args()

    root_directory = Path(args.root_directory)
    config_file = Path(args.config_file)
    is_update_run = args.is_update_run
    index_method = args.index_method
    bert_mode = args.bert_mode
    ner_labels = args.ner_labels

    progress_queue = asyncio.Queue()

    download_task = DownloadTask(
        session_id="123456789",
        progress_queue=progress_queue
    )

    #download_task,root_directory,config_file,run_identifier
    asyncio.run(get_index(download_task,root_directory,config_file,index_method,is_update_run,bert_mode,ner_labels))  # 正确地运行异步主任务


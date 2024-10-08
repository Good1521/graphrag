# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
Indexing API for GraphRAG.

WARNING: This API is under development and may undergo changes in future releases.
Backwards compatibility is not guaranteed at this time.
"""

from graphrag.config import CacheType, GraphRagConfig

from .cache.noop_pipeline_cache import NoopPipelineCache
from .create_pipeline_config import create_pipeline_config
from .emit.types import TableEmitterType
from .progress import (
    ProgressReporter,
)
from .run import run_pipeline_with_config
from .typing import PipelineRunResult
import asyncio 
from pathlib import Path
import argparse
from .progress.types import ReporterType


async def build_index(
    download_task,
    config: GraphRagConfig,
    run_id: str = "",
    is_resume_run: bool = False,
    is_update_run: bool = False,
    memory_profile: bool = False,
    progress_reporter: ProgressReporter | None = None,
    emit: list[str] | None = None,
):
    """Run the pipeline with the given configuration.

    Parameters
    ----------
    config : PipelineConfig
        The configuration.
    run_id : str
        The run id. Creates a output directory with this name.
    is_resume_run : bool default=False
        Whether to resume a previous index run.
    is_update_run : bool default=False
        Whether to update a previous index run.
    memory_profile : bool
        Whether to enable memory profiling.
    progress_reporter : ProgressReporter | None default=None
        The progress reporter.
    emit : list[str] | None default=None
        The list of emitter types to emit.
        Accepted values {"parquet", "csv"}.

    Returns
    -------
    list[PipelineRunResult]
        The list of pipeline run results
    """
    if is_resume_run and is_update_run:
        msg = "Cannot resume and update a run at the same time."
        raise ValueError(msg)

    pipeline_config = create_pipeline_config(config)

    # print(pipeline_config)
    pipeline_cache = (
        NoopPipelineCache() if config.cache.type == CacheType.none is None else None
    )
    outputs: list[PipelineRunResult] = []
    async for output in run_pipeline_with_config(
        download_task,
        pipeline_config,
        run_id=run_id,
        memory_profile=memory_profile,
        cache=pipeline_cache,
        progress_reporter=progress_reporter,
        emit=([TableEmitterType(e) for e in emit] if emit is not None else None),
        is_resume_run=is_resume_run,
        is_update_run=is_update_run,
    ):
        # show_json(stats_jaon,num_workflows)

        outputs.append(output)
        if progress_reporter:
            if output.errors and len(output.errors) > 0:
                progress_reporter.error(output.workflow)
            else:
                progress_reporter.success(output.workflow)
            progress_reporter.info(str(output.result))
                

    return outputs



def display_pipeline_results(outputs: list[PipelineRunResult]):
    for output in outputs:
        if output.result is not None and not output.result.empty:
            # 处理结果
            print(output.result)
        else:
            print("No results for this output")



async def get_index(download_task,root_directory,config_file,run_identifier,is_update_run):
    from graphrag.config import load_config
    from .progress.load_progress_reporter import load_progress_reporter

    # 指定根目录和配置文件路径
    # root_directory = Path('/home/turing/graphragtest/ragtest')
    # config_file = 'settings.yaml'
    # run_identifier = 'api_test07'
    # reporter = None   # rich, print, none

    # 调用函数加载配置
    config = load_config(root_directory, config_filepath=config_file)

    # 如果需要，可以加载进度报告器
    # progress_reporter = load_progress_reporter(ReporterType.RICH )
    progress_reporter = None

    # 使用配置
    outputs = await build_index(
        download_task=download_task,
        config=config,
        run_id=run_identifier,
        is_resume_run=False,
        is_update_run=is_update_run,
        memory_profile=False,
        progress_reporter=progress_reporter,
        emit=None
    )
    download_task.finish()
    # print("---------is_downloading----------",download_task.is_downloading)
    print("ok")
    return outputs
        
        

async def monitor_progress(progress_queue):
    # workflow_name = None  # 初始化
    # verb_name = None  # 初始化
    # now_workflows = None  # 初始化

    while True:
        progress = await progress_queue.get()
        # print(progress)

        if progress == "DONE":
            print()
            print(f"任务完成！ 完成度: {now_workflows}/{now_workflows}")
            break
        # show_json(progress)
        if progress["type"] == "workflow":
            workflow_name = progress["workflow_name"]
            now_workflows = progress["now_workflows"]
            num_workflows = progress["num_workflows"]
            print()
            print(f"开始运行workflow: {workflow_name} 正在运行第{now_workflows}个workflow 完成度为: {now_workflows-1}/{num_workflows}")


        elif progress["type"] == "verb":
            verb_name = progress["verb_name"]
            verb_len = progress["num_verb"]
            now_verb = progress["now_verb"]
            verb_time = progress["verb_time"]
            print(f"当前运行的workflow是: {workflow_name} 正在执行第 {now_verb} 个的verb: {verb_name} 上一个verb使用的时间是: {verb_time} 完成度为: {now_verb-1}/{verb_len}")

        elif progress["type"] == "workflow_time":
            workflow_overall = progress["workflow_overall"]
            total_runtime = progress["total_runtime"]
            print(f"完成workflow: {workflow_name} 的运行 该workflow的运行时间是 {workflow_overall} 总运行时间是 {total_runtime}")



async def main_task(root_directory,config_file,run_identifier, session_id, check_is_downloading):

    progress_queue = asyncio.Queue()  # 创建异步队列
    await asyncio.gather(
        get_index(progress_queue,root_directory,config_file,run_identifier, session_id, check_is_downloading),  # 启动异步任务
        monitor_progress(progress_queue)     # 启动监控进度
    )


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

    def complete_finished(self):
        self.complete_finish = True

    def graceful_stop(self):
        self.is_graceful_stop = True
        self.complete_finish = False




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, default='/home/turing/graphragtest/ragtest')
    parser.add_argument("--config_file", type=str, default='/home/turing/graphragtest/ragtest/settings.yaml')
    parser.add_argument("--run_identifier", type=str, default='api_test_169')
    parser.add_argument("--is_update_run", type=str, default=True)

    args = parser.parse_args()

    root_directory = args.root_directory
    config_file = args.config_file
    run_identifier = args.run_identifier
    is_update_run = args.is_update_run

    # asyncio.run(main_task(download_task,root_directory,config_file,run_identifier,is_update_run))  # 正确地运行异步主任务


    progress_queue = asyncio.Queue()

    download_task = DownloadTask(
        session_id="123456789",
        progress_queue=progress_queue
    )

    #download_task,root_directory,config_file,run_identifier
    asyncio.run(get_index(download_task,root_directory,config_file,run_identifier,is_update_run))  # 正确地运行异步主任务

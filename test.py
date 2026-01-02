# from rich.progress import Progress
# import time
#
# # Basic
# print(">>> Basic Usage <<<")
# with Progress() as progress:
#     task = progress.add_task("Processing...", total=100)
#     for i in range(100):
#         time.sleep(0.05)
#         progress.update(task, advance=1)
#
# # Multi-tasks
# print(">>> Multi Tasks <<<")
# with Progress() as progress:
#     task1 = progress.add_task("[cyan]Downloading...", total=100)
#     task2 = progress.add_task("[magenta]Processing...", total=200)
#
#     while not progress.finished:
#         progress.update(task1, advance=1)
#         progress.update(task2, advance=2)
#         time.sleep(0.05)
#
# # User Define
# from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
#
# print(">>> User Define <<<")
# with Progress(
#     "[progress.description]{task.description}",  # Task description
#     BarColumn(),                                 # process bar
#     TaskProgressColumn(),                        # percent
#     TimeElapsedColumn(),                         # time elapsed
#     TimeRemainingColumn()                        # time remaining
# ) as progress:
#     task = progress.add_task("Training model", total=1000)
#     for i in range(1000):
#         time.sleep(0.01)
#         progress.update(task, advance=1)
#
# # Refrash Info
# with Progress() as progress:
#     task = progress.add_task("Downloading", total=1000)
#
#     for i in range(1000):
#         if i == 500:
#             progress.update(task, description="Halfway done!")
#         progress.update(task, advance=1)
#         time.sleep(0.01)
#
# # Multi component
# from rich.live import Live
# from rich.table import Table
# # from rich.progress import Progress
# # import time
#
# print(">>> Multi Component <<<")
#
# progress = Progress()
# task1 = progress.add_task("Task 1", total=100)
# task2 = progress.add_task("Task 2", total=200)
#
# table = Table(title="Summary")
# table.add_column("Task")
# table.add_column("Status")
#
# with Live(progress, refresh_per_second=10):
#     for i in range(200):
#         if not progress.finished:
#             progress.update(task1, advance=1)
#             progress.update(task2, advance=1)
#         time.sleep(0.05)
#
#

from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
import time
import random

# 定义进度条
progress = Progress(
    "[progress.description]{task.description}",  # 描述
    BarColumn(),                                 # 进度条
    TaskProgressColumn(),                        # 百分比
    TimeElapsedColumn(),                         # 已用时间
    TimeRemainingColumn(),                       # 剩余时间
)

# 添加一个任务（训练）
task = progress.add_task("Training model", total=100)

# 定义表格显示指标
def make_metrics(epoch, loss1, loss2):
    table = Table.grid()
    table.add_row(f"[cyan]Epoch:[/] {epoch}")
    table.add_row(f"[green]Loss1:[/] {loss1:.4f}")
    table.add_row(f"[magenta]Loss2:[/] {loss2:.4f}")
    return Panel(table, title="Metrics", border_style="blue")

# 使用 Live 动态刷新
with Live(refresh_per_second=10) as live:
    for epoch in range(1, 101):
        time.sleep(0.05)
        progress.update(task, advance=1)

        loss1 = random.uniform(0.1, 1.0)
        loss2 = random.uniform(0.1, 1.0)

        live.update(
           Group(
                progress.get_renderable(),
                make_metrics(epoch, loss1, loss2)
                )
        )

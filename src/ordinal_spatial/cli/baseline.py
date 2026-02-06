"""
基线模型评估 CLI 入口。
CLI entry point for baseline evaluation.
"""


def main():
  """调用 run_baseline 脚本的主函数。"""
  from ordinal_spatial.scripts.run_baseline import main as _main
  _main()


if __name__ == "__main__":
  main()

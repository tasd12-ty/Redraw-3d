"""
基准数据集生成 CLI 入口。
CLI entry point for benchmark dataset generation.
"""


def main():
  """调用 build_benchmark 脚本的主函数。"""
  from ordinal_spatial.scripts.build_benchmark import main as _main
  _main()


if __name__ == "__main__":
  main()

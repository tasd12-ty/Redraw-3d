"""
基准数据集验证 CLI 入口。
CLI entry point for benchmark validation.
"""


def main():
  """调用 validate_benchmark 脚本的主函数。"""
  from ordinal_spatial.scripts.validate_benchmark import main as _main
  _main()


if __name__ == "__main__":
  main()

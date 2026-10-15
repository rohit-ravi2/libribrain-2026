import sys, os, runpy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import stretch_patch  # noqa
os.chdir(os.path.expanduser("~/projects/libribrain-2026")); sys.argv = ["scripts/train_v3.py"] + sys.argv[1:]
runpy.run_path("scripts/train_v3.py", run_name="__main__")

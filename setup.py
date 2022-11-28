from cx_Freeze import setup, Executable
import sys

buildOptions = dict(packages = ["bpy","mediapipe", "torch", "torchvision", "scipy", "tensorflow", "tensorflow.keras", "tensorflow", "mediapipe","torchvision", "tensorflow.keras"],  # 1
	excludes = [],
    includes=["torchvision",])

exe = [Executable("main.py")]

# 3
setup(
    name='head detection',
    version = '0.1',
    author = "me",
    description = "description ",
    options = dict(build_exe = buildOptions),
    executables = exe
)
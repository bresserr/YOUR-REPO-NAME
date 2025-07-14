from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vr-video-analyzer",
    version="1.0.0",
    author="VR Video Analyzer Team",
    author_email="info@vrvideoanalyzer.com",
    description="A comprehensive VR video analysis tool with body part detection and funscript generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vrvideoanalyzer/vr-video-analyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "tensorrt": [
            "tensorrt>=8.0.0",
            "pycuda>=2022.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "vr-video-analyzer=vr_video_analyzer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "vr_video_analyzer": [
            "static/*",
            "templates/*",
            "models/*",
        ],
    },
    keywords="vr video analysis body detection funscript tensorrt cuda machine learning",
    project_urls={
        "Bug Reports": "https://github.com/vrvideoanalyzer/vr-video-analyzer/issues",
        "Source": "https://github.com/vrvideoanalyzer/vr-video-analyzer",
        "Documentation": "https://vrvideoanalyzer.readthedocs.io/",
    },
)
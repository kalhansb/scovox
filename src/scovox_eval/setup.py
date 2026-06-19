from setuptools import find_packages, setup

setup(
    name="scovox_eval",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
    ],
    extras_require={
        "mesh": ["scikit-image", "open3d"],
        "eval": ["scikit-learn", "trimesh"],
        "ros": ["rclpy", "sensor_msgs"],
        "plot": ["matplotlib"],
        "dev": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "mesh-extract = scovox_eval.mesh_extract:main",
            "compute-fscore = scovox_eval.metrics.fscore:main",
            "compute-chamfer = scovox_eval.metrics.chamfer:main",
            "compute-miou = scovox_eval.metrics.miou:main",
            "compute-ece = scovox_eval.metrics.ece:main",
            "compute-eig = scovox_eval.metrics.eig:main",
            "compute-semantic-ece = scovox_eval.metrics.semantic_ece:main",
            "compute-semantic-brier = scovox_eval.metrics.semantic_brier:main",
            "compute-reliability-diagram = scovox_eval.metrics.semantic_reliability_diagram:main",
            "compute-runtime-stats = scovox_eval.metrics.compute_runtime_stats:main",
            "compute-unknown-ambiguous = scovox_eval.metrics.unknown_vs_ambiguous:main",
            "pointcloud-to-npz = scovox_eval.pointcloud_to_npz:main",
            "render-qualitative-compare = scovox_eval.render_qualitative_compare:main",
            "replica-replay = scovox_eval.replica_replay_node:main",
            "semantickitti-replay = scovox_eval.semantickitti_replay_node:main",
            "scenenet-replay = scovox_eval.scenenet_replay_node:main",
        ],
    },
    description="Evaluation and experiment toolkit for SCovox mapping",
    license="Apache-2.0",
)

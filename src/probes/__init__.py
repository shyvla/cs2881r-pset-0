"""Run-once diagnostics. Each probe answered one question whose conclusion is
now frozen into config.py. Dependencies point one way: probes import the core
modules (config, hooks, loaders, scoring); nothing in the live pipeline
(run.py, analyze.py) imports a probe.

Run from src/ as modules:  python -m probes.calibrate --tiny
"""

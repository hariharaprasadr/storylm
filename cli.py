#!/usr/bin/env python3
"""TinyLM Unified CLI: Generate stories, test KV-cache, serve the web visualizer, or inspect the model."""
from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import subprocess
import sys
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def cmd_story(args):
    """Run interactive story generation."""
    cmd = [sys.executable, os.path.join(ROOT, "tinylm/story.py")]
    if args.world:
        cmd.extend(["--world", args.world])
    if args.mode:
        cmd.extend(["--mode", args.mode])
    if args.place:
        cmd.extend(["--place", args.place])
    if args.theme:
        cmd.extend(["--theme", args.theme])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.temperature is not None:
        cmd.extend(["--temperature", str(args.temperature)])
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    if args.raw:
        cmd.append("--raw")
    if args.list:
        cmd.append("--list")
    return subprocess.run(cmd, cwd=ROOT).returncode


def cmd_test(args):
    """Run KV-cache equivalence and speed benchmark test."""
    cmd = [sys.executable, os.path.join(ROOT, "tinylm/test_cache.py")]
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    return subprocess.run(cmd, cwd=ROOT).returncode


def cmd_trace(args):
    """Re-export mathematical forward pass trace into web/trace_data.js."""
    cmd = [sys.executable, os.path.join(ROOT, "training/export_trace.py")]
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    return subprocess.run(cmd, cwd=ROOT).returncode


def cmd_web(args):
    """Serve the interactive web explainer and open in browser."""
    port = args.port
    web_dir = os.path.join(ROOT, "web")
    os.chdir(web_dir)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # quiet logging

    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            url = f"http://localhost:{port}/index.html"
            print(f"Serving TinyLM Web Explainer at {url}")
            print("Press Ctrl+C to stop.")
            if not args.no_browser:
                webbrowser.open(url)
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        return 0


def cmd_info(args):
    """Print model checkpoint information and dataset overview."""
    print("=" * 65)
    print("  TinyLM 1.67M Parameter Nano Language Model")
    print("=" * 65)
    vocab_path = os.path.join(ROOT, "data/vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            v = json.load(f)
            print(f"Vocabulary:     {len(v.get('tokens', []))} tokens")

    for ckpt_name in ("checkpoints/embedded_big.pt", "checkpoints/embedded.pt"):
        p = os.path.join(ROOT, ckpt_name)
        if os.path.exists(p):
            size_mb = os.path.getsize(p) / (1024 * 1024)
            print(f"Checkpoint:     {ckpt_name} ({size_mb:.2f} MB)")

    data_dir = os.path.join(ROOT, "data/processed")
    if os.path.exists(data_dir):
        print("\nDatasets:")
        for item in sorted(os.listdir(data_dir)):
            full = os.path.join(data_dir, item)
            size_mb = os.path.getsize(full) / (1024 * 1024)
            print(f"  - {item:<20} {size_mb:>8.2f} MB")
    print("=" * 65)
    return 0


def main():
    parser = argparse.ArgumentParser(description="TinyLM: Nano Model CLI & Web Visualizer")
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # story
    p_story = subparsers.add_parser("story", aliases=["generate"], help="Generate a character story")
    p_story.add_argument("--world", default="bluey", help="World: bluey, gabby, one, amelia, junie")
    p_story.add_argument("--mode", default="bedtime", help="Mode: bedtime, adventure, funny, kindness")
    p_story.add_argument("--place", default=None, help="Place inside chosen world")
    p_story.add_argument("--theme", default="help", help="Theme: help, share, brave, try")
    p_story.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    p_story.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    p_story.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint")
    p_story.add_argument("--raw", action="store_true", help="Print raw draft tokens")
    p_story.add_argument("--list", action="store_true", help="List available characters & places")

    # test
    p_test = subparsers.add_parser("test", help="Verify KV-cache equivalence & latency")
    p_test.add_argument("--checkpoint", default=None)

    # trace
    p_trace = subparsers.add_parser("trace", help="Export forward pass trace for web visualizer")
    p_trace.add_argument("--checkpoint", default=None)

    # web
    p_web = subparsers.add_parser("web", help="Start local server for interactive web visualizer")
    p_web.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    p_web.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")

    # info
    subparsers.add_parser("info", help="Print model and dataset summary")

    args = parser.parse_args()
    if args.subcommand in ("story", "generate"):
        return cmd_story(args)
    elif args.subcommand == "test":
        return cmd_test(args)
    elif args.subcommand == "trace":
        return cmd_trace(args)
    elif args.subcommand == "web":
        return cmd_web(args)
    elif args.subcommand == "info":
        return cmd_info(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

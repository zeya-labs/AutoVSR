#!/usr/bin/env python3
"""Compatibility entrypoint for synthetic image-to-netlist evaluation."""

from __future__ import annotations

from eval_vlm_netlist_level0 import main


if __name__ == "__main__":
    raise SystemExit(main())

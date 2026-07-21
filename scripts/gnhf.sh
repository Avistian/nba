#!/usr/bin/env bash
exec "$(git rev-parse --show-toplevel)/flow/bin/gnhf" "$@"

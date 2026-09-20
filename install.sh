#!/usr/bin/env bash
# Wrapper kept at the repository root so `bash install.sh` keeps working;
# the installer itself lives in installers/install.sh.
exec bash "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/installers/install.sh" "$@"

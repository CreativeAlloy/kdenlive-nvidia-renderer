#!/usr/bin/env bash
# Adds "Kdenlive Nvidia Renderer" to your application menu (and "Open With" for .kdenlive files).
set -e
DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
mkdir -p "$HOME/.local/share/applications"
cat > "$HOME/.local/share/applications/kdenlive-nvidia-renderer.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=Kdenlive Nvidia Renderer
Comment=Render Kdenlive projects on an NVIDIA GPU
Exec="$DIR/run.sh" %f
Icon=kdenlive
Terminal=false
Categories=AudioVideo;Video;
MimeType=application/x-kdenlive;
DESK
update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
echo "Installed: $HOME/.local/share/applications/kdenlive-nvidia-renderer.desktop"

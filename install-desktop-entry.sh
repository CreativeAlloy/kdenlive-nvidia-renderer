#!/usr/bin/env bash
# Adds "Kdenlive Nvidia Renderer" to your application menu, to "Open With" for .kdenlive files,
# and a "Render with Nvidia" item to the right-click menu of .kdenlive files in Nemo (Linux Mint).
set -e
DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

# Kdenlive's icon is named differently depending on how Kdenlive was installed.
ICON=org.kde.kdenlive
if [ -e /usr/share/icons/hicolor/scalable/apps/kdenlive.svg ]; then
    ICON=kdenlive
fi

APPS="$HOME/.local/share/applications"
mkdir -p "$APPS"
cat > "$APPS/kdenlive-nvidia-renderer.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=Kdenlive Nvidia Renderer
Comment=Render Kdenlive projects on an NVIDIA GPU
Exec="$DIR/run.sh" %f
Icon=$ICON
Terminal=false
Categories=AudioVideo;Video;
MimeType=application/x-kdenlive;
DESK
update-desktop-database "$APPS" 2>/dev/null || true
echo "Installed: $APPS/kdenlive-nvidia-renderer.desktop"

ACTIONS="$HOME/.local/share/nemo/actions"
mkdir -p "$ACTIONS"
cat > "$ACTIONS/kdenlive-nvidia-renderer.nemo_action" <<ACTION
[Nemo Action]
Name=Render with Nvidia
Comment=Open this project in Kdenlive Nvidia Renderer
Exec="$DIR/run.sh" %F
Icon-Name=$ICON
Selection=s
Extensions=kdenlive;
Quote=double
ACTION
echo "Installed: $ACTIONS/kdenlive-nvidia-renderer.nemo_action"

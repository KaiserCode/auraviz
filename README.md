# AuraViz

Audio visualization plugin for VLC Media Player 3.0.x (Windows 64-bit).

21 visual presets that react to your music in real time — spectrum bars, nebulas, fire, galaxies, and more. Switch presets instantly while music plays, no track restart needed.

![VLC Screenshot](screenshots/preview.png)

## Features

- **21 Visual Presets** — from classic spectrum bars to full-screen generative art
- **Real-Time Audio Analysis** — proper DFT frequency analysis with automatic gain control
- **Live Preset Switching** — change presets with arrow buttons, applies instantly without restarting the track
- **Control Panel UI** — dark themed HTML interface integrated into VLC's View menu
- **Auto Cycle Mode** — automatically rotates through all presets on bass drops
- **Persistent Settings** — saves your last preset and engine choice between sessions

## Presets

| # | Name | Style |
|---|------|-------|
| 0 | Auto Cycle | Rotates presets on bass hits |
| 1 | Spectrum Bars | Classic frequency bars |
| 2 | Waveform | Oscilloscope-style wave |
| 3 | Circular | Radial frequency ring |
| 4 | Particles | Audio-driven particle system |
| 5 | Nebula | Cosmic gas clouds |
| 6 | Plasma | Flowing plasma field |
| 7 | Tunnel | Infinite tunnel zoom |
| 8 | Kaleidoscope | Fractal mirror patterns |
| 9 | Lava Lamp | Metaball blob simulation |
| 10 | Starburst | Radial star burst |
| 11 | Electric Storm | Lightning bolts |
| 12 | Ripple Pool | Water ripple distortion |
| 13 | Fractal Warp | Warped noise fields |
| 14 | Spiral Galaxy | Rotating galaxy arms |
| 15 | Glitch Matrix | Digital glitch grid |
| 16 | Aurora Borealis | Northern lights waves |
| 17 | Pulse Grid | Pulsing grid lines |
| 18 | Fire | Rising flame simulation |
| 19 | Diamond Rain | Falling diamond particles |
| 20 | Vortex | Twisting vortex tunnel |

## Installation

### From Release (Recommended)

1. Download `libauraviz_plugin.dll` from the [latest release](../../releases)
2. Copy it to your VLC plugins folder:
   ```
   C:\Program Files\VideoLAN\VLC\plugins\visualization\
   ```
3. Copy `auraviz_menu.lua` to:
   ```
   C:\Program Files\VideoLAN\VLC\lua\extensions\
   ```
4. Restart VLC

### From GitHub Actions

1. Go to the **Actions** tab
2. Click the latest successful build
3. Download the `auraviz-vlc-plugin-win64` artifact
4. Extract and copy `libauraviz_plugin.dll` to the plugins folder above
5. Copy `auraviz_menu.lua` to the extensions folder above
6. Restart VLC

## Usage

1. Open VLC and play any audio file
2. Go to **View > AuraViz > Show Settings** to open the control panel
3. Use the `<<` and `>>` buttons to cycle through presets — changes apply instantly
4. The first time you enable it, skip or restart your track to activate the visualizer

### Control Panel

The control panel shows the current status (active/off), engine mode (CPU/OpenGL), and current preset. Use the `Switch to OpenGL` / `Switch to CPU` button to toggle rendering engines (requires track restart to take effect).

### Submenu

Under **View > AuraViz** you'll find:
- **Show Settings** — opens the control panel
- **Buy me a Coffee** — opens the support page in your browser

## How It Works

### Audio Analysis

AuraViz uses a real DFT (Discrete Fourier Transform) to extract 64 log-spaced frequency bands from 30Hz to 15kHz. A Hann window prevents spectral leakage, and automatic gain control (AGC) continuously tracks the recent peak magnitude so the visualization adapts to both quiet acoustic tracks and loud electronic music. Bass, mid, and treble channels use fast attack and medium release smoothing so every beat comes through clearly.

### Rendering

Presets 1-3 (Spectrum, Waveform, Circular) render at full resolution. All other presets render at half resolution and upscale to reduce CPU load while maintaining smooth framerates. The plugin runs its own render thread at ~50fps independent of VLC's audio pipeline.

### Live Preset Switching

The Lua control panel writes preset changes to VLC's global config via `vlc.config.set()`. The C plugin polls `config_GetInt()` every frame (~20ms) and switches presets instantly when it detects a change — no track restart required.

## Building from Source

The project uses GitHub Actions for automated builds. To build locally with MSYS2/MinGW64:

```bash
# Install dependencies
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-pkg-config make p7zip curl

# Download VLC 3.0.23 SDK
curl -L -o vlc.7z "https://download.videolan.org/vlc/last/win64/vlc-3.0.23-win64.7z"
7z x vlc.7z "vlc-3.0.23/sdk/*" -r -y
mv vlc-3.0.23/sdk ./vlc-sdk

# Set up pkg-config paths
SDK_PATH=$(cd vlc-sdk && pwd)
sed -i "s|^prefix=.*|prefix=${SDK_PATH}|g" vlc-sdk/lib/pkgconfig/*.pc
export PKG_CONFIG_PATH="${SDK_PATH}/lib/pkgconfig"

# Compile CPU plugin
gcc -Wall -O2 -shared -static-libgcc \
    -DMODULE_STRING=\"auraviz\" \
    $(pkg-config --cflags vlc-plugin) \
    -o libauraviz_plugin.dll \
    auraviz.c \
    $(pkg-config --libs vlc-plugin) \
    -lopengl32 -lm -lws2_32 \
    -Wl,-Bstatic -lpthread -Wl,-Bdynamic

# Compile OpenGL plugin
gcc -Wall -O2 -shared -static-libgcc \
    -DMODULE_STRING=\"auraviz_gl\" \
    $(pkg-config --cflags vlc-plugin) \
    -o libauraviz_gl_plugin.dll \
    auraviz_gl.c \
    $(pkg-config --libs vlc-plugin) \
    -lopengl32 -lm -lws2_32 \
    -Wl,-Bstatic -lpthread -Wl,-Bdynamic
```

## Project Structure

```
auraviz.c              # Main CPU visualization plugin
auraviz_gl.c           # OpenGL rendering plugin
auraviz_menu.lua       # Lua extension — control panel and menu UI
.github/workflows/
  build.yml            # GitHub Actions CI workflow
```

## Requirements

- VLC Media Player 3.0.x (64-bit Windows)
- Windows 10 or later

## Support the Project

If you enjoy AuraViz, consider buying me a coffee:

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffab40?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/davekaiser)

## License

GNU LGPL 2.1+

"""Font loading and text rasterisation (pygame.font) with bundled OFL fonts.

Text is always rasterised white with per-pixel alpha; renderers tint it with the
requested colour. Rendered surfaces are cached (LRU) because most HUD strings are
static between frames.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import pygame

from dragon_ball_powers.config import FONTS_DIR

log = logging.getLogger(__name__)

# kind -> (bundled font file or None, system font fallbacks in order of preference)
FONT_KINDS: dict[str, tuple[str | None, tuple[str, ...]]] = {
    "ui": ("Rajdhani-SemiBold.ttf", ("bahnschrift", "segoeui", "arial", "dejavusans")),
    "bold": ("Rajdhani-Bold.ttf", ("bahnschrift", "segoeuisemibold", "arial", "dejavusans")),
    "display": ("Bangers-Regular.ttf", ("impact", "arialblack", "bahnschrift", "dejavusans")),
    "mono": (None, ("consolas", "cascadiamono", "couriernew", "dejavusansmono")),
}


class FontManager:
    def __init__(self, fonts_dir: Path = FONTS_DIR, max_cached_surfaces: int = 600) -> None:
        if not pygame.font.get_init():
            pygame.font.init()
        self._fonts_dir = Path(fonts_dir)
        self._fonts: dict[tuple[str, int], pygame.font.Font] = {}
        self._surfaces: OrderedDict[tuple[str, int, str], pygame.Surface] = OrderedDict()
        self._max_cached = max_cached_surfaces
        self._warned: set[str] = set()

    def font(self, size: int, kind: str = "ui") -> pygame.font.Font:
        size = max(6, int(round(size)))
        key = (kind, size)
        cached = self._fonts.get(key)
        if cached is not None:
            return cached
        bundled, fallbacks = FONT_KINDS.get(kind, FONT_KINDS["ui"])
        font: pygame.font.Font | None = None
        if bundled:
            path = self._fonts_dir / bundled
            if path.is_file():
                try:
                    font = pygame.font.Font(str(path), size)
                except (OSError, pygame.error) as exc:
                    self._warn_once(kind, f"Could not load bundled font {path}: {exc}")
            else:
                self._warn_once(kind, f"Bundled font missing: {path}; using system font")
        if font is None:
            try:
                font = pygame.font.SysFont(",".join(fallbacks), size, bold=(kind == "bold"))
            except (OSError, pygame.error):
                font = None
        if font is None:
            font = pygame.font.Font(None, size)  # pygame's built-in default font
        self._fonts[key] = font
        return font

    def measure(self, text: str, size: int, kind: str = "ui") -> tuple[int, int]:
        if not text:
            return (0, self.font(size, kind).get_height())
        w, h = self.font(size, kind).size(text)
        return (int(w), int(h))

    def line_height(self, size: int, kind: str = "ui") -> int:
        return int(self.font(size, kind).get_linesize())

    def render(self, text: str, size: int, kind: str = "ui") -> pygame.Surface:
        """Rasterise ``text`` in white with per-pixel alpha (cached)."""
        key = (text, max(6, int(round(size))), kind)
        surface = self._surfaces.get(key)
        if surface is not None:
            self._surfaces.move_to_end(key)
            return surface
        font = self.font(key[1], kind)
        surface = font.render(text if text else " ", True, (255, 255, 255))
        if surface.get_bitsize() != 32 or not (surface.get_flags() & pygame.SRCALPHA):
            converted = pygame.Surface(surface.get_size(), pygame.SRCALPHA, 32)
            converted.blit(surface, (0, 0))
            surface = converted
        self._surfaces[key] = surface
        if len(self._surfaces) > self._max_cached:
            self._surfaces.popitem(last=False)
        return surface

    def _warn_once(self, kind: str, message: str) -> None:
        if kind not in self._warned:
            self._warned.add(kind)
            log.warning(message)

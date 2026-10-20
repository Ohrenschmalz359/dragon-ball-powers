# Fonts

The HUD, the settings panel and the attack callouts use two open-source font
families. They are bundled with the app so it looks the same on every PC; if a
file is missing, `dragon_ball_powers/fonts.py` falls back to a Windows system font.

| File | Family | Used for |
|------|--------|----------|
| `Bangers-Regular.ttf` | [Bangers](https://github.com/google/fonts/tree/main/ofl/bangers) | Display text such as attack callouts and titles (`kind="display"`) |
| `Rajdhani-Medium.ttf` | [Rajdhani](https://github.com/google/fonts/tree/main/ofl/rajdhani) by the Indian Type Foundry | Additional weight (bundled; `fonts.py` currently uses SemiBold and Bold) |
| `Rajdhani-SemiBold.ttf` | Rajdhani | Regular HUD and settings text (`kind="ui"`) |
| `Rajdhani-Bold.ttf` | Rajdhani | Emphasised HUD text (`kind="bold"`) |

## Licence

Both families are licensed under the **SIL Open Font License, Version 1.1**. The
full licence texts, with the copyright notices, ship next to the fonts and must
stay with them:

* [`Bangers-OFL.txt`](Bangers-OFL.txt): Copyright 2010 The Bangers Project Authors
  (<https://github.com/googlefonts/bangers>)
* [`Rajdhani-OFL.txt`](Rajdhani-OFL.txt): Copyright (c) 2014, Indian Type Foundry

The OFL allows the fonts to be used, bundled and redistributed with software, free of
charge. They may not be sold on their own, the licence and copyright notices must
accompany them, and modified versions must stay under the OFL (see the licence for
the details).

## Source

The font files and licence texts come from the Google Fonts repository:
<https://github.com/google/fonts> (folders `ofl/bangers` and `ofl/rajdhani`).

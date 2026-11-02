# Arcade Window

A 2012 iMac running Linux Mint sits at a first-floor window in a Newcastle arcade and watches
the view: a glass dome, two brick walls, two satellite dishes and a wedge of sky.

`collector/collector.py` runs on the iMac, reads the camera a few times a second and writes small
public derivatives into `data/`, which are pushed here every few minutes. `index.html` is the page,
served by GitHub Pages.

Panels: Bird Register (motion events boxed, named from the library shelves, naive re-identification),
Shadow Clock (the sunlit wall as an ASCII grid every five minutes), Weather for the Dome (palettes
sampled from the sky, wall, dome and brick), Jangle (an edge drawing of the view disturbed by its light).

Raw frames never leave the iMac.

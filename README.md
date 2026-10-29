# Tithe map: automatic cells + click-to-assign tracer

Put these in the folder that holds your map JPEG and CSV (no other .jpg files in that folder, or pass --image).

1. Cut the map into candidate cells (about 15 s; use your ORIGINAL full-size scan, not a downsized copy):
       python autotrace.py "Stanstead Abbots DSA4_97_2 whole map (3).jpg"
   This writes auto_cells.json and auto_preview.jpg. A ready-made auto_cells.json for the half-size
   (2174x2953) coordinates is included, so you can skip this step to try things out.
   Needs: numpy opencv-python scipy scikit-image

2. Assign plot numbers:
       python tracer_app.py --image "Stanstead Abbots DSA4_97_2 whole map (3).jpg" --csv "Tithe Plot owners and tenants.csv"
   then open http://127.0.0.1:8123
   - click a cell = give it the plot number in the sidebar, then the next plot is selected
   - shift-click several cells, then Enter = one plot made of several cells (woods)
   - Draw polygon = click the corners yourself (hatched strips, anything with no matching cell)
   - click a finished plot, drag white dots to adjust; double-click an edge to add a dot; alt-click a dot to delete it
   - Generate maps = writes interactive.html (hover map) and labelled_map.png

Image sharpness: polygons are stored at the working size (scan x 0.5), but the browser is shown your FULL-SIZE scan
(drawn into the same coordinates), so numbers stay readable when you zoom. A light copy is used while zoomed out so
panning stays smooth. --display-max N caps the longest side of the displayed image (default 8000 px).
"Generate maps" writes interactive.html plus interactive_map_full.jpg - keep the two files together. If you want one
single file instead, start the tracer with --embed-full (the html then becomes about as big as the scan).

Your existing tithe_polygons.json is read as-is (plot numbers become text so 123a, 453b etc. work).
--scale must match between the two steps (default 0.5 for both).
Tuning in autotrace.py (dict P at the top): lo/hi = line sensitivity (lower = fainter lines, more noise),
merge_paper = how eagerly over-split fields are re-joined.

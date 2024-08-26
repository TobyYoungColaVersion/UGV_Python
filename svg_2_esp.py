from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM
 
# Set input and output file paths
input_file = "a.svg"
output_file = "a.eps"
 
# Convert SVG to EPS
drawing = svg2rlg(input_file)
renderPDF.drawToFile(drawing, output_file)
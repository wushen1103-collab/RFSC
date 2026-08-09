Create a publication-ready technical framework figure for an ESWA molecular-machine-learning paper.

Visual style:
- White background, no large title.
- Match the visual language of the manuscript figures: Times New Roman text, pale blue-gray borders, thin dry lines, ample whitespace.
- Use only this palette: blue #4C78B6, sky #67A9DC, purple #8172B3, red #C94F53, gray #8B8E91, orange #D9852B, ink #2D3642, spine #BFC9D2, grid/faint fill #E9EEF3.
- Rounded white boxes with a narrow colored left accent bar.
- Thin arrows, 0.75–1 px, with small arrowheads. No icons, no gradients, no dark background.
- Do not add decorative elements.

Content and layout:
- Single left-to-right framework schematic.
- Left column has four boxes:
  1. "responses" with a red cross/forbidden mark to indicate not used during correction.
  2. "frozen scalar predictions"
  3. "frozen direct-pair predictions"
  4. "query structures"
- Middle-left column has two boxes:
  1. "MMP graph and active edges"
  2. "ordered pair graph and incidence map"
- Middle-right column has two boxes:
  1. "BPS node projection"
  2. "Hodge edge projection"
- Right column has two boxes:
  1. "response-free risk gate"
  2. "corrected output or identity fallback"
- Bottom audit rail: "protocol hashes · label firewall · split audit"

Connections:
- frozen scalar predictions + query structures -> MMP graph and active edges -> BPS node projection -> response-free risk gate.
- frozen direct-pair predictions + query structures -> ordered pair graph and incidence map -> Hodge edge projection -> response-free risk gate.
- response-free risk gate -> corrected output or identity fallback.
- bottom audit rail has dashed orange audit arrows to both projection operators.
- responses box must not connect to the inference pipeline.

Constraints:
- Spell all labels exactly as written.
- No extra explanatory paragraphs inside the figure.
- No panel letters.
- Keep all text horizontal and readable at journal print size.

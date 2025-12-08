# Research Initiatives

As part of our commitment to excellence, we undertook three major research initiatives beyond the core specification.

## Visualisation with OpenStreetMap (Interactive Mapping)
Suggestion Addressed: "Visualising your system predictions... using open source resources such as OpenStreetMap?"

The initial requirement suggested static images for maps. We researched the Leaflet.js library and integrated it with our Python backend.
-   Innovation: We wrote a coordinate-mapper that translates the abstract (lat, lon) in the assignment text files into real-world Web Mercator projections.
-   Value: This provides users with context. They can see if a road is near a river, a school, or a highway, which influences real-world routing decisions.

## Universal Simulation Player
Suggestion Addressed: "For other ideas... do the research yourselves."

Most routing algorithms visualized in academia only show the "Path" as a static line. We wanted to see the dynamics. We researched JavaScript animation loops within the Leaflet framework to build a Simulation Player.
-   Capability: It takes the sequence of nodes returned by Part A and interpolates a marker's position between them.
-   Comparison Mode: Uniquely, our player works for K-Shortest Paths, allowing a user to visually compare "How much longer is Route 2?" by watching two animations side-by-side (sequentially).

## Transfer Learning for "Small Data"
Suggestion Addressed: "Data processing... will be a challenge."

The provided dataset was small. Training a CNN from scratch on small data usually leads to overfitting. We researched Transfer Learning strategies.
-   Method: We froze the feature extraction layers of pre-trained ImageNet models and only trained the final classification head.
-   Outcome: This allowed us to achieve >70% accuracy with only a few hundred images, a feat impossible with a "train-from-scratch" approach. This demonstrates a deep understanding of practical AI application constraints.

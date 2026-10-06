# Licences (per source; each subset keeps its own licence)

## VCTK 0.92: baselines B01-B08 and every clip derived from them
CSTR VCTK Corpus 0.92, (c) University of Edinburgh, **CC BY 4.0**, https://datashare.ed.ac.uk/handle/10283/3443.
Cite: Veaux, Yamagishi, MacDonald (2017), https://doi.org/10.7488/ds/2645.
Changes made: first utterances concatenated, silence trimmed to 0.3 s margins, resampled to 22.05 kHz mono, one gain change;
altered variants are derivative works (CC BY 4.0, same attribution). Obtained through the Hugging Face mirror `sanchit-gandhi/vctk`.

## Fillers
All fillers ("uh", "um", drawls) are built from the SAME speaker's own recorded vowels (no synthetic or foreign audio).

## Planned subsets (not yet in the dataset; add a section here when ingested)
- LibriVox Gettysburg Address readers: public domain (USA).
- JFK Rice University speech 12 Sep 1962: US government work, public domain (JFK Library).
- Speech Accent Archive (GMU): non-commercial; mirrors list CC BY-NC-SA, confirm on the site. Altered clips inherit the same terms.
- Svarah (AI4Bharat): CC BY 4.0. Tried as a baseline source and dropped (speakers' own disfluencies); may return as a natural-disfluency test set.
